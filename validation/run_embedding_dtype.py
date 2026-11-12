import ctypes
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

BASE = "a616886cacf86701514f0cb1d6be760f967481c0"
HEAD = "8b51310726bc1321bde5652296bf7010d05f997b"
SOURCE = "bitsandbytes/nn/modules.py"
TESTS = "tests/test_modules.py"
ROOT = Path.cwd()
subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", BASE, HEAD], check=True)
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
import bitsandbytes
import torch
from bitsandbytes.cextension import lib
assert isinstance(lib._lib, ctypes.CDLL), type(lib)
native_path = Path(lib._lib._name)
print("Native library:", str(native_path), hashlib.sha256(native_path.read_bytes()).hexdigest(), flush=True)
print("PyTorch:", torch.__version__, "bitsandbytes:", bitsandbytes.__version__, flush=True)
environment = dict(os.environ, PYTHONPATH=str(ROOT), BNB_TEST_DEVICE="cpu", OMP_NUM_THREADS="2")
def run(label, expression):
    result = subprocess.run([sys.executable, "-m", "pytest", TESTS, "-q", "-k", expression,
                             f"--junitxml={label}.xml"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    Path(f"{label}.log").write_text(result.stdout)
    print(result.stdout, flush=True)
    suites = ElementTree.parse(f"{label}.xml").getroot()
    totals = {k: sum(int(s.get(k, 0)) for s in suites.iter("testsuite"))
              for k in ("tests", "failures", "errors", "skipped")}
    totals["returncode"] = result.returncode
    print(label, totals, flush=True)
    return totals
fixed = (ROOT / SOURCE).read_bytes()
try:
    (ROOT / SOURCE).write_bytes(subprocess.check_output(["git", "show", f"{BASE}:{SOURCE}"]))
    before = run("before", "quantized_embedding_dtype")
finally:
    (ROOT / SOURCE).write_bytes(fixed)
assert before == {"tests": 15, "failures": 15, "errors": 0, "skipped": 0, "returncode": 1}, before
after = run("after", "embedding")
assert after["returncode"] == 0 and after["failures"] == after["errors"] == 0, after
assert after["tests"] - after["skipped"] >= 77, after
for path in (SOURCE, TESTS):
    assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", f"{HEAD}:{path}"])
summary = {"head":HEAD, "base":BASE, "torch":torch.__version__, "bitsandbytes":bitsandbytes.__version__,
           "native_library": str(native_path), "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
           "before":before, "after":after}
Path("validation_result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
