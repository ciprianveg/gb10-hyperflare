"""Owned-memory CUDA clearing diagnostic; GPU execution requires a separate release.

No model, communicator, capture or export is constructed. --check is CPU-only.
The interval calculation mirrors capture.rs::storage_ranges and
reachability.rs::merge_ranges; it is not a substitute for a driver result.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path


HOST_BYTES = (71983104, 44453888)


def storage_ranges(ranges):
    intervals = []
    for address, size in ranges:
        if address <= 0 or size <= 0 or address + size > (1 << 64) - 1:
            raise ValueError("nonnull positive storage range without u64 overflow required")
        intervals.append((address, address + size))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def check_ranges():
    assert storage_ranges([(20, 10), (10, 10)]) == [(10, 30)]
    assert storage_ranges([(10, 5), (20, 5)]) == [(10, 15), (20, 25)]
    assert storage_ranges([(12, 3), (10, 10)]) == [(10, 20)]
    assert storage_ranges([]) == []
    for bad in ([(0, 1)], [(1, 0)], [((1 << 64) - 1, 1)]):
        try:
            storage_ranges(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)


def emit(**record):
    print(json.dumps(record, sort_keys=True), flush=True)


def host_matches(address, size, value):
    # Bound temporary CPU storage while checking every byte, including padding.
    for offset in range(0, size, 1 << 20):
        count = min(1 << 20, size - offset)
        if C.string_at(address + offset, count) != bytes([value]) * count:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--extension", type=Path)
    args = parser.parse_args()
    check_ranges()
    if args.check:
        emit(cpu_interval_cases="passed", gpu_execution=False)
        return
    if args.extension is None:
        parser.error("--extension is required for the separately released GPU invocation")
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        raise RuntimeError("separate Ops GPU release required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("this diagnostic requires explicitly assigned local GPU 0")
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        path = Path(os.environ[name]).resolve(strict=True)
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise RuntimeError(f"{name} must be an existing writable private cache")

    from fixture_owners import FixtureOwners
    from multi_roce_fixture import candidate_import
    extension = candidate_import(args.extension)
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.module_images import ModuleImages

    torch.cuda.set_device(0)
    context, _ = extension.current_context()
    results = []
    with FixtureOwners(lambda owners: extension.export_synchronize(context, owners)) as ownership:
        try:
            images = ownership.enter(ModuleImages())
            storage = ownership.enter(CoreStorage(rank=0, world=1, device=0, local_ranks=1))
            hosts = [storage.host_allocate(f"clear-host-{i}", (size,), torch.uint8)
                     for i, size in enumerate(HOST_BYTES)]
            device = ownership.hold(torch.empty(4096, dtype=torch.uint8, device="cuda:0"))
            mappings = list(storage.mappings.values())
            assert any(m.address <= device.data_ptr()
                       and device.data_ptr() + device.numel() <= m.address + m.nbytes
                       for m in mappings)
            regions = [dict(kind="mapped_host", address=h.address, requested=h.nbytes,
                            extent=h.extent, end=h.address + h.extent) for h in hosts]
            regions.append(dict(kind="device", address=device.data_ptr(), requested=device.numel(),
                                extent=device.numel(), end=device.data_ptr() + device.numel()))
            original = [(r["address"], r["extent"]) for r in regions]
            merged = storage_ranges(original)
            emit(package=extension.__file__, regions=regions,
                 device_backing_mappings=[vars(m) for m in mappings],
                 individual_intervals=[(a, a + n) for a, n in original],
                 exporter_merged_intervals=merged,
                 host_registrations_adjacent=hosts[0].address + hosts[0].extent == hosts[1].address)
            # _Cuda is the actual storage driver wrapper; retain integer statuses
            # rather than discarding the failing call's range in an exception.
            cuda = storage.cuda
            memset = cuda.lib.cuMemsetD8_v2
            memset.argtypes = (C.c_uint64, C.c_ubyte, C.c_size_t)
            memset.restype = C.c_int
            for mode, intervals in (
                ("individual", [(a, a + n) for a, n in original]),
                ("exporter_merged", merged),
            ):
                for start, end in intervals:
                    covered = [r for r in regions if start <= r["address"] and r["end"] <= end]
                    assert sum(r["extent"] for r in covered) == end - start
                    cuda.call("cuCtxSynchronize")
                    for h in hosts:
                        C.memset(h.address, 0xA5, h.extent)
                    device.fill_(0xA5)
                    torch.cuda.synchronize()
                    assert all(host_matches(h.address, h.extent, 0xA5) for h in hosts)
                    assert bool((device.cpu() == 0xA5).all())
                    record = dict(mode=mode, start=start, length=end-start,
                                  crossed_boundaries=sorted({r["end"] for r in covered
                                                             if r["end"] < end}),
                                  initialized_nonzero=True)
                    status = memset(start, 0, end-start)
                    record["cuda_status"] = status
                    results.append(record)
                    emit(call=record)
                    if status:
                        try:
                            cuda.check("cuMemsetD8_v2", status)
                        except RuntimeError as error:
                            record["error"] = str(error)
                    # A synchronization failure stops further cases and enters the
                    # existing native drain guard; it is not automatically retried.
                    cuda.call("cuCtxSynchronize")
                    if status == 0:
                        checks = []
                        for h in hosts:
                            value = 0 if start <= h.address and h.address + h.extent <= end else 0xA5
                            checks.append(host_matches(h.address, h.extent, value))
                        value = 0 if start <= device.data_ptr() and device.data_ptr() + device.numel() <= end else 0xA5
                        checks.append(bool((device.cpu() == value).all()))
                        record["full_readback_and_untouched_regions_exact"] = all(checks)
                        emit(result=record)
                        assert all(checks), "successful driver clear failed complete byte readback"
                    else:
                        emit(result=record)
            assert storage.error is None
        finally:
            ownership.hold(dict(locals()))
    assert ownership.drained
    emit(native_owner_lease_drained=True, storage_and_images_exited=True, results=results,
         qualification="diagnostic only; no exporter repair or distributed qualification")


if __name__ == "__main__":
    main()
