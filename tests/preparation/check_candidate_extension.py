"""CPU-only candidate import and actual export metadata validation.

Valid transport descriptions stop at an intentionally invalid serving metadata
object, before capture_rank can touch CUDA. This is not capture/export proof.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import ctypes

from multi_roce_fixture import candidate_import


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    expected = args.extension.resolve(strict=True)
    flare_heliopause = candidate_import(expected)
    from flare_heliopause import flare_heliopause as native_module
    assert Path(native_module.__file__).resolve() == expected
    for name in ("export_capture", "export_synchronize", "current_context", "NativeModuleRecorder"):
        assert getattr(flare_heliopause, name) is getattr(native_module, name)
    library = ctypes.CDLL(str(expected))
    symbols = ("flare_core_error", "flare_core_snapshot", "flare_core_string_free",
               "flare_core_initialize", "flare_core_allocate", "flare_core_free",
               "flare_core_transient_allocate", "flare_core_transient_free",
               "flare_core_retention_begin", "flare_core_retention_finish", "flare_core_retention_abort")
    for symbol in symbols:
        assert getattr(library, symbol)  # Resolve only; never call allocator/CUDA functions on CPU.
    assert "libcuda.so" not in Path("/proc/self/maps").read_text()
    lane = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "candidate_roce_metadata", lane/"ignition/preparation/roce_transport.py")
    metadata = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata)

    def native(world, base):
        return dict(proxy_abi=14,world_size=world,rank=0,
                    region=dict(address=base,nbytes=8192),
                    counters=dict(address=base+0x4000,nbytes=4096),
                    slot_bytes=16,pipe_bytes=16,ctrl_offset=4096,varlen_bytes=0)

    def reject(transports, message):
        request = dict(path="unused-cpu-smoke",context=0,rank=0,world_size=16,
                       graphs=[],allocations=[],transports=transports,
                       execution=None,server=[],rebuild_weights=None,core_data=None)
        try:
            flare_heliopause.export_capture(json.dumps(request), object())
        except ValueError as error:
            assert message in str(error), str(error)
        else:
            raise AssertionError("CPU metadata guard did not reject before capture")
        assert "libcuda.so" not in Path("/proc/self/maps").read_text()

    tp = metadata.description(native(16,0x10000),"tp",list(range(16)),world=16,rank=0)
    dcp = metadata.description(native(8,0x20000),"dcp-a",list(range(8)),world=16,rank=0)
    reject([native(16,0x10000)],"serving metadata must be an object")
    reject([tp,dcp],"serving metadata must be an object")
    reject([tp,tp],"IDs must be nonempty and unique")
    reject([tp,native(16,0x20000)],"legacy cannot be mixed")
    wrong = dict(dcp,members=list(range(8,16)))
    reject([tp,wrong],"membership/local rank differs")
    print(json.dumps(dict(extension=str(expected),package=flare_heliopause.__file__,
                          native_exports_checked=len(native_module.__all__),allocator_symbols=list(symbols),
                          checks=5,passed=True,
                          cuda_library_loaded=False,capture_executed=False)))


if __name__ == "__main__":
    main()
