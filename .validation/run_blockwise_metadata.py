import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

BASE = "a616886cacf86701514f0cb1d6be760f967481c0"
HEAD = "229715336ad8ec8a17dd546821843a9909ab5f73"
SOURCE = "bitsandbytes/backends/cpu/ops.py"
TESTS = "tests/test_cpu_blockwise_metadata.py"
ROOT = Path.cwd()
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])

import bitsandbytes
import torch
from bitsandbytes.cextension import lib

assert isinstance(lib._lib, ctypes.CDLL), type(lib)
native_path = Path(lib._lib._name)
native_sha = hashlib.sha256(native_path.read_bytes()).hexdigest()
print("Native CPU library:", native_path, native_sha, flush=True)
print("PyTorch:", torch.__version__, "bitsandbytes:", bitsandbytes.__version__, flush=True)
environment = dict(os.environ, PYTHONPATH=str(ROOT), BNB_TEST_DEVICE="cpu", OMP_NUM_THREADS="2")


def run(label, paths, expression=None):
    command = [sys.executable, "-m", "pytest", *paths, "-q", f"--junitxml={label}.xml"]
    if expression:
        command += ["-k", expression]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    suites = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(t.get(k, 0)) for t in suites.iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    failed = sorted({case.get("name").split("[")[0] for case in suites.iter("testcase")
                     if case.find("failure") is not None})
    totals["failed_functions"] = failed
    print(label, totals, flush=True)
    return totals


fixed = (ROOT / SOURCE).read_bytes()
try:
    (ROOT / SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", [TESTS])
finally:
    (ROOT / SOURCE).write_bytes(fixed)
after = run("after", [TESTS])
existing = run("existing", ["tests/test_ops.py::TestInt8BlockwiseQuantOps",
                           "tests/test_ops.py::TestNonContiguousInputs"], "blockwise")
summary = {"base": BASE, "head": HEAD, "torch": torch.__version__,
           "bitsandbytes": bitsandbytes.__version__, "native_library": str(native_path),
           "native_sha256": native_sha, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
assert before["returncode"] == 1 and before["errors"] == 0, before
assert "test_cpu_blockwise_quantization_uses_codebook_values" in before["failed_functions"], before
assert "test_cpu_blockwise_dequantization_reads_strided_metadata" in before["failed_functions"], before
assert "test_cpu_adam8bit_continuation_with_strided_state" in before["failed_functions"], before
assert after["returncode"] == after["failures"] == after["errors"] == after["skipped"] == 0, after
assert after["tests"] == 31, after
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
