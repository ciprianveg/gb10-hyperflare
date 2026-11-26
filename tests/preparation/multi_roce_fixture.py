"""Weightless TP16/DCP8 producer; capture/export requires a separate GPU release.

No launcher or directory publisher is included. The storage owner must create
the root and all rank directories using the retained publication helper.
"""
import argparse
from datetime import timedelta
import importlib.util
import json
import os
from pathlib import Path
import sys

from multi_roce_capture import capture_mixed
from fixture_owners import FixtureOwners
from multi_roce_reference import (
    CHUNK_BYTES, GROUPS, HOSTS, SCHEDULE, VALID_BYTES, WORLD,
    gather_reference, history_output_reference, logical_accounting, communicator_accounting,
    members, reduce_reference, source,
)

LANE = Path(__file__).resolve().parents[2]
STAGING = LANE.parent


def retained_fixture():
    """Load only the retained CPU admission/export helper module."""
    location = STAGING/"collective-replay-split"
    sys.path.insert(0,str(location))
    try:
        spec = importlib.util.spec_from_file_location("retained_split_fixture",location/"fixture.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(location))


def admit(args):
    return retained_fixture().admit_rank_directory(
        args.directory/f"rank-{args.rank}",args.directory,args.rank,args.export_id,
        args.directory_owner_uid,args.directory_owner_gid)


def candidate_import(extension):
    import sysconfig

    expected = extension.resolve(strict=True)
    package = expected.parent
    initializer = package / "__init__.py"
    if (package.name != "flare_heliopause"
            or expected.name != "flare_heliopause" + sysconfig.get_config_var("EXT_SUFFIX")
            or not initializer.is_file()):
        raise RuntimeError("candidate requires the standard private flare_heliopause package layout")
    sys.path.insert(0,str(package.parent))
    sys.path.insert(0,str(LANE))
    import flare_heliopause
    import ignition
    if Path(flare_heliopause.__file__).resolve() != initializer.resolve():
        raise RuntimeError("baseline package won import resolution")
    from flare_heliopause import flare_heliopause as native
    if Path(native.__file__).resolve() != expected:
        raise RuntimeError("baseline nested extension won import resolution")
    for name in native.__all__:
        if getattr(flare_heliopause, name) is not getattr(native, name):
            raise RuntimeError(f"package export is not the candidate native API: {name}")
    if Path(ignition.__file__).resolve().parent != LANE/"ignition":
        raise RuntimeError("fixture must use the candidate preparation source")
    return flare_heliopause


def expected_bytes(operation, rank):
    if operation.kind == "reduce":
        return reduce_reference(operation,rank)
    if operation.kind == "gather":
        return gather_reference(operation,rank)
    return history_output_reference(operation,rank)


def cpu_plan(rank):
    return dict(rank=rank,host=HOSTS[rank],memberships=[list(WORLD),list(GROUPS[rank//8])],
                protocol_accounting=communicator_accounting(rank),
                operations=[dict(index=op.index,communicator=op.communicator,kind=op.kind,
                                 shape=[op.rows,op.columns],logical=logical_accounting(op,rank))
                            for op in SCHEDULE])


def initialize_gloo(args, dist):
    """Bind Gloo to this rank's reviewed fabric address before creating groups."""
    from ignition.preparation.host_ranks import _use_interface

    if type(args.rank) is not int or not 0 <= args.rank < len(HOSTS):
        raise ValueError("fixture rank must select a reviewed fabric host")
    address = HOSTS[args.rank]
    _use_interface(address)
    interface = os.environ["GLOO_SOCKET_IFNAME"]
    if interface == "lo" or address.startswith("127."):
        raise RuntimeError("fixture Gloo interface must not be loopback")
    print(json.dumps(dict(gloo_rank=args.rank, local_address=address,
                          local_interface=interface)), flush=True)
    dist.init_process_group("gloo",init_method=f"tcp://{args.rendezvous}",rank=args.rank,
                            world_size=16,timeout=timedelta(minutes=5))


def prepare(args):
    # No imports that can initialize CUDA precede owner directory admission.
    folder = admit(args)
    extension = candidate_import(args.extension)
    # Deliberately malformed serving metadata terminates in the real CPU codec.
    sentinel = dict(path="unused",context=0,rank=args.rank,world_size=16,
                    graphs=[],allocations=[],transports=[],execution=None,
                    server=[],rebuild_weights=None,core_data=None)
    try:
        extension.export_capture(json.dumps(sentinel),object())
    except ValueError as error:
        if "serving metadata must be an object" not in str(error):
            raise
    else:
        raise RuntimeError("CPU codec sentinel did not stop before capture")
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    import torch.distributed as dist
    from ignition.components.collectives import PreparedAllReduce, PreparedAllGather, PreparedVarlenGather
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.runtime_core import export_model
    from ignition.preparation.tuning import RecordedPreparationSession
    # Reuse the existing small signed fixture contract, not a model recipe.
    sys.path.insert(0,str(STAGING/"collective-replay-pipe/helios/tests"))
    from collective_core_fixture import _configuration
    for key in ("FLARE_ROCE_VARLEN_CHUNK_BYTES","FLARE_ROCE_PIPE_CHUNK_BYTES","B12X_ROCE_LANES"):
        if key in os.environ:
            raise RuntimeError(f"ambient transport override is not permitted: {key}")
    torch.set_num_threads(1)
    initialize_gloo(args, dist)
    graphs, sessions, runtimes = {}, {}, {}
    ownership = None
    try:
        # All global ranks create the same subgroup list in the same order.
        subgroups = [dist.new_group(ranks=list(group),backend="gloo",timeout=timedelta(minutes=5))
                     for group in GROUPS]
        group = subgroups[args.rank//8]
        peers = [None]*16
        dist.all_gather_object(peers,dict(export_id=args.export_id,schedule=cpu_plan(0)["operations"]))
        if any(peer != peers[0] for peer in peers):
            raise RuntimeError("ranks disagree on fixture identity/schedule")
        torch.cuda.set_device(0)
        device = torch.device("cuda",0)
        context, _ = extension.current_context()
        # export_synchronize's native OwnerLease pins this entire guard on
        # uncertain drain; no enclosing storage/image context may then unwind.
        with FixtureOwners(lambda owners: extension.export_synchronize(context,owners)) as ownership:
            ownership.hold((graphs,sessions,runtimes,subgroups,dist.group.WORLD))
            ownership.after_drain(lambda: [graph.reset() for graph in graphs.values()])
            ownership.after_drain(lambda: [session.close() for session in sessions.values()])
            ownership.after_drain(lambda: [runtime.close() for runtime in runtimes.values()])
            ownership.after_contexts(dist.destroy_process_group)
            images = ownership.enter(ModuleImages())
            storage = ownership.enter(CoreStorage(rank=args.rank,world=16,device=0,local_ranks=1))
            try:
                coordinators = dict(tp=CollectivePreparation(dist.group.WORLD),dcp=CollectivePreparation(group))
                for name, process_group, membership in (
                    ("tp",dist.group.WORLD,WORLD),("dcp",group,GROUPS[args.rank//8]),
                ):
                    identifier = "tp" if name == "tp" else ("dcp-a" if args.rank < 8 else "dcp-b")
                    runtimes[name] = create_roce_runtime(
                        process_group,device,max_tokens=17,pipe_tokens=17,hidden_size=256,
                        gather_bytes=3*256*2,varlen_bytes=0 if name == "tp" else CHUNK_BYTES+16,
                        core_storage=storage,communicator_id=identifier,members=list(membership))
                    description = runtimes[name].native_description()
                    if (description["world_size"] != len(membership)
                            or description["rank"] != membership.index(args.rank)
                            or description["rails"] != 2 or description["ring"] or description["ring_open"]):
                        raise RuntimeError("fixture requires independent two-rail mesh communicators")
                    if name == "dcp" and runtimes[name]._varlen_chunk_packs*16 != CHUNK_BYTES:
                        raise RuntimeError("installed variable gather chunk geometry differs")
                    sessions[name] = RecordedPreparationSession(
                        device=device,checkpoint="weightless-multi-roce",
                        workload_schema="tp16-dcp8-mixed-bf16-v1",numerical_policy="exact-small-bf16",
                        coordinator=coordinators[name],compile_workers=1)
                tensors, calls, checks = {}, [], []
                for op in SCHEDULE:
                    data = bytearray(source(op,args.rank))
                    src = torch.frombuffer(data,dtype=torch.uint8).clone().to(device).view(torch.bfloat16)
                    src = src.view(op.rows,op.columns)
                    expected = expected_bytes(op,args.rank)
                    shape = (op.rows,op.columns) if op.kind == "reduce" else (
                        (op.rows,op.columns*len(members(op,args.rank))) if op.kind == "gather"
                        else (8,op.columns))
                    out = torch.zeros(shape,dtype=torch.bfloat16,device=device)
                    source_name, output_name = f"source-{op.index}", f"output-{op.index}"
                    tensors[source_name], tensors[output_name] = src,out
                    kwargs = dict(runtime=runtimes[op.communicator],session=sessions[op.communicator],
                                  coordinator=coordinators[op.communicator],name=f"mixed.{op.index}.{op.kind}")
                    if op.kind == "selection-history":
                        counts = torch.tensor(VALID_BYTES,dtype=torch.int64,device=device)
                        tensors["history-counts"] = counts
                        operation = PreparedVarlenGather(out,counts,**kwargs)
                        local = args.rank%8
                        count = VALID_BYTES[local]//2
                        def call(operation=operation,src=src,out=out,local=local,count=count):
                            out.zero_()
                            out[local,:count].copy_(src.view(-1)[:count])
                            operation()
                    else:
                        cls = PreparedAllReduce if op.kind == "reduce" else PreparedAllGather
                        operation = cls(src,out,**kwargs)
                        def call(operation=operation,src=src):
                            operation(src)
                    calls.append(call)
                    checks.append(dict(source=source_name,output=output_name,input=list(data),expected=list(expected)))
                # Each coordinator sees only its own declarations. Keep both sessions
                # alive and unfrozen until both jobs and compiler pools have drained.
                sessions["tp"].tune(show_progress=False)
                dist.barrier()
                sessions["dcp"].tune(show_progress=False)
                dist.barrier()
                if any(getattr(session,"_startup_collectives",None) is not None for session in sessions.values()):
                    raise RuntimeError("pure RoCE operations unexpectedly need PCIe startup registry")
                def mixed():
                    for call in calls:
                        call()
                marker = torch.zeros(1,dtype=torch.int32,device=device)
                def boundary():
                    marker.add_(0)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    mixed()
                stream.synchronize()
                dist.barrier()
                names = ("init","decode","prefill64","prefill128","restore","snapshot","resume")
                signatures = {name:GraphSignature("prefill" if name.startswith("prefill") else name,
                    1,int(name[7:]) if name.startswith("prefill") else 1) for name in names}
                for name in names:
                    dist.barrier()
                    graphs[name] = capture_mixed(mixed if name == "decode" else boundary,
                        tp_runtime=runtimes["tp"],dcp_runtime=runtimes["dcp"],
                        tp_session=sessions["tp"],dcp_session=sessions["dcp"],stream=stream,
                        storage=storage,owners=(tensors,calls,marker))
                for session in sessions.values():
                    session.freeze()
                state = storage.device_allocate("continuations",(4,32),torch.uint8,
                                                group="STATE_ROWS",maximum=4,units=4)
                pages = storage.device_allocate("pages",(16,16),torch.uint8,
                                                group="KV_PAGES",maximum=16,units=16)
                buffers = dict(frame_index=torch.zeros(1,dtype=torch.int32,device=device),
                    submission_sequence=torch.zeros(1,dtype=torch.int64,device=device),
                    completion_records=torch.zeros((2,1,32),dtype=torch.int64,device=device),
                    state_pool=state,pages=pages,marker=marker,
                    memory_capacity=torch.tensor([4,16],device=device))
                records = buffers["completion_records"]
                records[...,4] = -1
                records[...,12:14] = -1
                records[...,15] = 1
                records[...,16:] = -1
                buffers.update({name:value.view(torch.uint8) for name,value in tensors.items()})
                prepared = PreparedExecution(device=0,graphs=graphs,buffers=buffers,signatures=signatures,
                    max_active=1,export_zero_ranges=tuple((value.data_ptr(),value.numel()*value.element_size())
                        for name,value in tensors.items() if name != "history-counts"),
                    resources=(runtimes,sessions,tensors,calls,storage,stream))
                prepared.server_config = _configuration(folder,signatures)
                prepared.server_config["model_id"] = "weightless-tp16-dcp8"
                prepared.core_data = None
                descriptions = [registration.native_description() for registration in storage.transports]
                transport_plan = [dict(id=d["communicator_id"],members=d["members"],
                    region_address=d["region"]["address"],region_bytes=d["region"]["nbytes"],
                    counters_address=d["counters"]["address"],counters_bytes=d["counters"]["nbytes"])
                    for d in descriptions]
                torch.cuda.synchronize()
                dist.barrier()
                export_model(prepared,images,storage,folder/"rank.json",options={},
                             bundle_only=args.rank != 0,sources=True)
                replay = dict(rank=args.rank,export_id=args.export_id,transports=transport_plan,checks=checks,
                    protocol_accounting=communicator_accounting(args.rank),
                    preparation_pid=os.getpid(),
                    preparation_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]),
                    boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
                with (folder/"replay-plan.json").open("x") as file:
                    json.dump(replay,file)
                dist.barrier()
            finally:
                # Include tensors/stream/recorders and partial locals even if an
                # operation failed before insertion into its resource dictionary.
                ownership.hold(dict(locals()))
    finally:
        # Once the guard exists, only its confirmed-drain path may release the
        # groups. Before that point no fixture graph/storage has been created.
        if ownership is None:
            original = sys.exception()
            try:
                dist.destroy_process_group()
            except BaseException as error:
                if original is None:
                    raise
                original.add_note(f"pre-storage group cleanup failed: {error!r}")


def combine(args):
    candidate_import(args.extension)
    from ignition.preparation.core_file import combine as publish
    sys.path.insert(0,str(LANE/"helios/tests"))
    from signing import publisher
    paths = []
    for rank in range(16):
        folder = args.directory/f"rank-{rank}"
        plan = json.loads((folder/"replay-plan.json").read_text())
        if plan["rank"] != rank or plan["export_id"] != args.export_id:
            raise ValueError("rank export identity mismatch")
        paths.append(folder/"rank.json")
    publish(args.directory/"core",paths,[],publisher=publisher(),cluster=(16,args.export_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("plan","prepare","combine"))
    parser.add_argument("--rank",type=int,default=0)
    parser.add_argument("--directory",type=Path)
    parser.add_argument("--export-id")
    parser.add_argument("--extension",type=Path)
    parser.add_argument("--rendezvous")
    parser.add_argument("--directory-owner-uid",type=int)
    parser.add_argument("--directory-owner-gid",type=int)
    parser.add_argument("--exclusive-fleet-confirmed",action="store_true")
    args = parser.parse_args()
    if not 0 <= args.rank < 16:
        parser.error("rank must be in 0..15")
    if args.mode == "plan":
        print(json.dumps(cpu_plan(args.rank),indent=2))
        return
    if not args.exclusive_fleet_confirmed or args.directory is None or not args.export_id or args.extension is None:
        parser.error("requires reviewed directory/export identity/private extension and explicit fleet release")
    if args.mode == "prepare":
        if not args.rendezvous or args.directory_owner_uid is None or args.directory_owner_gid is None:
            parser.error("prepare requires rendezvous and assigned directory owner")
        prepare(args)
    else:
        combine(args)


if __name__ == "__main__":
    main()
