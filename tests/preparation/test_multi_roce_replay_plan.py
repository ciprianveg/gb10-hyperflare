"""CPU process tests for the actual native tool's pre-CUDA plan admission."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


def plan(rank):
    group = rank // 8
    return {
        "rank": rank, "export_id": "cpu-admission-only", "preparation_pid": 0,
        "preparation_start_ticks": 0, "boot_id": "not-used-by-cpu-check",
        "transports": [
            dict(id="tp", members=list(range(16)), region_address=0x10000,
                 region_bytes=8192, counters_address=0x14000, counters_bytes=4096),
            dict(id=f"dcp-{'a' if group == 0 else 'b'}", members=list(range(group*8, group*8+8)),
                 region_address=0x20000, region_bytes=8192,
                 counters_address=0x24000, counters_bytes=4096),
        ],
        "checks": [dict(source="source-0", output="output-0",
                        input=[0, 0], expected=[0, 0])],
    }


class NativePlanAdmission(unittest.TestCase):
    binary = None

    def run_plan(self, value, valid):
        with tempfile.TemporaryDirectory(prefix="multi-roce-plan-", dir="/tmp/opencode") as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(value))
            result = subprocess.run(
                [self.binary, "--plan", str(path), "--check-plan"],
                text=True, capture_output=True, timeout=10,
            )
        self.assertEqual(result.returncode == 0, valid, result.stdout + result.stderr)
        if valid:
            self.assertTrue(json.loads(result.stdout)["cpu_plan_valid"])

    def test_all_global_ranks_use_their_own_subgroup(self):
        for rank in range(16):
            with self.subTest(rank=rank):
                self.run_plan(plan(rank), True)

    def test_missing_duplicate_and_mismatched_members_fail_before_bootstrap(self):
        cases = []
        value = plan(0)
        value["transports"].pop()
        cases.append(value)
        value = plan(0)
        value["transports"][1] = copy.deepcopy(value["transports"][0])
        cases.append(value)
        value = plan(8)
        value["transports"][1]["members"] = list(range(8))
        cases.append(value)
        value = plan(0)
        value["transports"][1]["members"].reverse()
        cases.append(value)
        for value in cases:
            self.run_plan(value, False)

    def test_overflow_overlap_alignment_and_empty_regions_fail(self):
        for key, number in [("region_address", 2**64-4096), ("region_address", 0x10000),
                            ("region_address", 0x20001), ("region_bytes", 0)]:
            value = plan(0)
            value["transports"][1][key] = number
            self.run_plan(value, False)

    def test_reference_buffers_cannot_alias_by_name_or_be_empty(self):
        value = plan(0)
        value["checks"][0]["output"] = "source-0"
        self.run_plan(value, False)
        value = plan(0)
        value["checks"][0]["input"] = []
        self.run_plan(value, False)

    def test_device_counters_need_int32_alignment_not_page_alignment(self):
        value = plan(0)
        value["transports"][1]["counters_address"] += 4
        self.run_plan(value, True)
        value["transports"][1]["counters_address"] += 2
        self.run_plan(value, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    args = parser.parse_args()
    NativePlanAdmission.binary = str(args.binary.resolve(strict=True))
    unittest.main(argv=[__file__], verbosity=2)
