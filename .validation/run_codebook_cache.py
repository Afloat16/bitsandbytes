import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree

BASE = "a616886cacf86701514f0cb1d6be760f967481c0"
HEAD = "5fa7276f1f2590b7e27419c02ddd05f25b5b1612"
SOURCE = "csrc/cpu_ops.cpp"
TESTS = "tests/test_cpu_codebook_cache.py"
ROOT = Path.cwd()
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
environment = dict(os.environ, PYTHONPATH=str(ROOT), BNB_TEST_DEVICE="cpu", OMP_NUM_THREADS="2")


def build(label):
    subprocess.run(["cmake", "--build", "build-cpu", "--parallel", "2"], check=True)
    check = """import ctypes, hashlib, json
from pathlib import Path
import torch
import bitsandbytes
from bitsandbytes.cextension import lib
assert isinstance(lib._lib, ctypes.CDLL), type(lib)
path = Path(lib._lib._name)
print(json.dumps({"torch": torch.__version__, "bitsandbytes": bitsandbytes.__version__, "library": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
"""
    process = subprocess.run([sys.executable, "-c", check], check=True, capture_output=True, text=True, env=environment)
    provenance = json.loads(process.stdout.strip().splitlines()[-1])
    Path(f"{label}_native.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(label, provenance, flush=True)
    return provenance


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
    totals["failed_functions"] = sorted({case.get("name").split("[")[0] for case in suites.iter("testcase")
                                        if case.find("failure") is not None})
    print(label, totals, flush=True)
    return totals


fixed = (ROOT / SOURCE).read_bytes()
try:
    (ROOT / SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before_native = build("before")
    before = run("before", [TESTS])
finally:
    (ROOT / SOURCE).write_bytes(fixed)
after_native = build("after")
after = run("after", [TESTS])
existing = run("existing", ["tests/test_ops.py::TestInt8BlockwiseQuantOps",
                           "tests/test_ops.py::TestNonContiguousInputs"], "blockwise")
summary = {"base": BASE, "head": HEAD, "before_native": before_native,
           "after_native": after_native, "before": before, "after": after, "existing": existing}
Path("validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
assert before_native["sha256"] != after_native["sha256"], summary
assert before["returncode"] == 1 and before["errors"] == 0, before
assert "test_cpu_quantization_refreshes_changed_codebook_interior" in before["failed_functions"], before
assert "test_updated_cpu_codebook_persists_consistent_linear_weights" in before["failed_functions"], before
assert after["returncode"] == after["failures"] == after["errors"] == after["skipped"] == 0, after
assert after["tests"] == 21, after
assert existing["returncode"] == existing["failures"] == existing["errors"] == 0, existing
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
