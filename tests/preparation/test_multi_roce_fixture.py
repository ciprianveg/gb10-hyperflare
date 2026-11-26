"""CPU producer-plan and retained worker-directory admission checks."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from multi_roce_fixture import admit, cpu_plan, expected_bytes
from multi_roce_reference import SCHEDULE, VALID_BYTES


class FixtureAdmission(unittest.TestCase):
    def test_rank_directory_validation_precedes_gpu_imports(self):
        with tempfile.TemporaryDirectory(prefix="multi-roce-admission-",dir="/tmp/opencode") as tmp:
            root = Path(tmp)/"unit-export"
            directory = root/"artifacts"
            directory.mkdir(parents=True)
            for rank in range(16):
                leaf = directory/f"rank-{rank}"
                leaf.mkdir()
                args = SimpleNamespace(directory=directory,rank=rank,export_id=root.name,
                                       directory_owner_uid=os.getuid(),directory_owner_gid=os.getgid())
                self.assertEqual(admit(args),leaf)
                self.assertEqual(list(leaf.iterdir()),[])
                (leaf/"occupied").write_text("owned-test-data")
                with self.assertRaises(ValueError):
                    admit(args)
                (leaf/"occupied").unlink()
                args.directory_owner_uid = os.getuid()+1
                with self.assertRaises(ValueError):
                    admit(args)
            self.assertNotIn("torch",sys.modules)
            self.assertNotIn("flare_heliopause",sys.modules)

    def test_existing_admission_rejects_symlinks_and_namespace_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="multi-roce-admission-",dir="/tmp/opencode") as tmp:
            root = Path(tmp)/"unit-export"
            directory = root/"artifacts"
            directory.mkdir(parents=True)
            target = root/"elsewhere"
            target.mkdir()
            (directory/"rank-0").symlink_to(target,target_is_directory=True)
            args = SimpleNamespace(directory=directory,rank=0,export_id=root.name,
                                   directory_owner_uid=os.getuid(),directory_owner_gid=os.getgid())
            with self.assertRaises(ValueError):
                admit(args)
            args.export_id = "different-export"
            with self.assertRaises(ValueError):
                admit(args)

    def test_producer_plan_and_reference_shapes_match_local_group(self):
        for rank in (0,7,8,15):
            plan = cpu_plan(rank)
            self.assertEqual(len(plan["memberships"][0]),16)
            self.assertEqual(len(plan["memberships"][1]),8)
            self.assertEqual(plan["protocol_accounting"]["dcp"]["dispatch_bytes"],18944+VALID_BYTES[rank%8])
            for op in SCHEDULE:
                result = expected_bytes(op,rank)
                scale = 1 if op.kind == "reduce" else (16 if op.communicator == "tp" else 8)
                self.assertEqual(len(result),op.rows*op.columns*2*scale)


if __name__ == "__main__":
    unittest.main()
