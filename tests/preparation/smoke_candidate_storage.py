"""Unrun single-GPU prerequisite check; no RoCE, model, capture or export.

Ops must release one device and preserve the qualified toolchain and private
TRITON_CACHE_DIR/TORCHINDUCTOR_CACHE_DIR before invoking this entry point.
"""
import argparse
import json
import os
from pathlib import Path

from fixture_owners import FixtureOwners
from multi_roce_fixture import candidate_import


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True,
                        help="nested native module in the private standard package")
    args = parser.parse_args()
    if os.environ.get("FLARE_RUN_CUDA_TESTS") != "1":
        raise RuntimeError("separate Ops single-device GPU release required")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not selected or "," in selected:
        raise RuntimeError("exactly one explicitly assigned GPU is required")
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        path = Path(os.environ[name]).resolve(strict=True)
        if not path.is_dir() or not os.access(path, os.W_OK):
            raise RuntimeError(f"{name} must be an existing writable private cache")
    extension = candidate_import(args.extension)
    from ignition import runtime_patches
    runtime_patches.apply()
    import torch
    # Import the actual fixture preparation path, not a mocked storage facade.
    from ignition.components.collectives import PreparedAllReduce, PreparedAllGather, PreparedVarlenGather
    from ignition.preparation.collectives import CollectivePreparation, create_roce_runtime
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.execution import GraphSignature, PreparedExecution
    from ignition.preparation.module_images import ModuleImages
    from ignition.preparation.runtime_core import export_model
    from ignition.preparation.tuning import RecordedPreparationSession

    torch.cuda.set_device(0)
    context, _ = extension.current_context()
    with FixtureOwners(lambda owners: extension.export_synchronize(context, owners)) as ownership:
        try:
            images = ownership.enter(ModuleImages())
            storage = ownership.enter(CoreStorage(rank=0, world=1, device=0, local_ranks=1))
            expected = torch.arange(1024, dtype=torch.int32, device="cpu")
            allocation = ownership.hold(torch.empty(1024, dtype=torch.int32, device="cuda:0"))
            allocation.copy_(expected)
            readback = allocation.cpu()
            assert torch.equal(readback, expected)
            address = allocation.data_ptr()
            assert any(m.address <= address and address + allocation.numel() * allocation.element_size()
                       <= m.address + m.nbytes for m in storage.mappings.values())
            assert storage.error is None
            print(json.dumps(dict(package=extension.__file__, rank=0, world=1, local_ranks=1,
                                  allocation_bytes=4096, address=address, readback_exact=True,
                                  scope="CoreStorage/CUPTI prerequisite only; no capture/export/RoCE")),
                  flush=True)
        finally:
            ownership.hold(dict(locals()))
    assert ownership.drained
    print("confirmed native OwnerLease drain; storage/image contexts exited", flush=True)


if __name__ == "__main__":
    main()
