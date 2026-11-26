"""Actual private package/codec imports and rejection checks in fresh CPU processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class CandidatePackageTests(unittest.TestCase):
    extension = None
    flat = None

    def child(self, code):
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[2],
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                                     "PYTHONDONTWRITEBYTECODE": "1"},
                                text=True, capture_output=True)
        print(result.stdout, end="", flush=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_standard_package_and_five_real_codec_checks(self):
        self.child("import sys,runpy\nsys.path.insert(0,'tests/preparation')\n"
                   f"sys.argv=['check_candidate_extension.py','--extension',{str(self.extension)!r}]\n"
                   "runpy.run_path('tests/preparation/check_candidate_extension.py',run_name='__main__')")

    def test_old_flat_layout_is_rejected_without_cuda(self):
        self.child("import sys\nfrom pathlib import Path\nsys.path.insert(0,'tests/preparation')\n"
                   "from multi_roce_fixture import candidate_import\n"
                   f"try: candidate_import(Path({str(self.flat)!r}))\n"
                   "except RuntimeError as error:\n"
                   " assert 'standard private' in str(error); print('expected_flat_rejection:',error)\n"
                   "else: raise AssertionError('flat extension accepted')\n"
                   "assert 'flare_heliopause' not in sys.modules\n"
                   "assert 'libcuda.so' not in Path('/proc/self/maps').read_text()")

    def test_preimported_installed_package_is_not_silently_replaced(self):
        self.child("import sys\nfrom pathlib import Path\nimport flare_heliopause\n"
                   "assert '/site-packages/' in flare_heliopause.__file__\n"
                   "sys.path.insert(0,'tests/preparation')\nfrom multi_roce_fixture import candidate_import\n"
                   f"try: candidate_import(Path({str(self.extension)!r}))\n"
                   "except RuntimeError as error:\n"
                   " assert 'baseline package' in str(error); print('expected_installed_rejection:',error)\n"
                   "else: raise AssertionError('installed package accepted')\n"
                   "assert 'libcuda.so' not in Path('/proc/self/maps').read_text()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--flat-extension", type=Path, required=True)
    args, rest = parser.parse_known_args()
    CandidatePackageTests.extension = args.extension.resolve(strict=True)
    CandidatePackageTests.flat = args.flat_extension.resolve(strict=True)
    print(json.dumps(dict(extension=str(CandidatePackageTests.extension), flat=str(CandidatePackageTests.flat))))
    unittest.main(argv=[sys.argv[0], *rest])
