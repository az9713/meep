import subprocess
import os
import sys
import time
import json

PYTHON = os.path.expanduser("~/miniforge3/envs/mp/bin/python")
MEEP_DIR = "/mnt/c/Users/simon/Downloads/meep-master"
EXAMPLES_DIR = os.path.join(MEEP_DIR, "python/examples")
TESTS_DIR = os.path.join(MEEP_DIR, "python/tests")
RESULTS_FILE = "/mnt/c/Users/simon/Downloads/meep-master/test_results.json"

os.environ["MPLBACKEND"] = "Agg"

results = {"examples": [], "tests": [], "summary": {}}

def run_file(filepath, timeout=180):
    name = os.path.basename(filepath)
    start = time.time()
    try:
        proc = subprocess.run(
            [PYTHON, filepath],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(filepath),
            env={**os.environ, "MPLBACKEND": "Agg"}
        )
        elapsed = round(time.time() - start, 2)
        status = "PASS" if proc.returncode == 0 else "FAIL"
        error = proc.stderr[-500:] if proc.returncode != 0 else ""
        print(f"  {status} {name} ({elapsed}s)", flush=True)
        return {"name": name, "status": status, "time": elapsed, "error": error}
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - start, 2)
        print(f"  TIMEOUT {name} ({elapsed}s)", flush=True)
        return {"name": name, "status": "TIMEOUT", "time": elapsed, "error": "exceeded timeout"}
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        print(f"  ERROR {name} ({elapsed}s)", flush=True)
        return {"name": name, "status": "ERROR", "time": elapsed, "error": str(e)[:200]}

print("=" * 60, flush=True)
print("RUNNING PYTHON EXAMPLES", flush=True)
print("=" * 60, flush=True)
examples = sorted([
    os.path.join(EXAMPLES_DIR, f) for f in os.listdir(EXAMPLES_DIR)
    if f.endswith(".py") and not f.startswith("__")
])
print(f"Found {len(examples)} example files", flush=True)
for fp in examples:
    results["examples"].append(run_file(fp, timeout=180))

adj_dir = os.path.join(EXAMPLES_DIR, "adjoint_optimization")
if os.path.isdir(adj_dir):
    adj = sorted([os.path.join(adj_dir, f) for f in os.listdir(adj_dir) if f.endswith(".py") and not f.startswith("__")])
    print(f"Found {len(adj)} adjoint examples", flush=True)
    for fp in adj:
        results["examples"].append(run_file(fp, timeout=300))

print(flush=True)
print("=" * 60, flush=True)
print("RUNNING PYTHON TESTS", flush=True)
print("=" * 60, flush=True)
tests = sorted([
    os.path.join(TESTS_DIR, f) for f in os.listdir(TESTS_DIR)
    if f.startswith("test_") and f.endswith(".py")
])
print(f"Found {len(tests)} test files", flush=True)
for fp in tests:
    results["tests"].append(run_file(fp, timeout=300))

all_r = results["examples"] + results["tests"]
p = sum(1 for r in all_r if r["status"] == "PASS")
f = sum(1 for r in all_r if r["status"] == "FAIL")
t = sum(1 for r in all_r if r["status"] == "TIMEOUT")
e = sum(1 for r in all_r if r["status"] == "ERROR")
total_time = round(sum(r["time"] for r in all_r), 1)
results["summary"] = {"total": len(all_r), "passed": p, "failed": f, "timeout": t, "errors": e, "total_time": total_time}

print(flush=True)
print("=" * 60, flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Total:   {len(all_r)}", flush=True)
print(f"Passed:  {p}", flush=True)
print(f"Failed:  {f}", flush=True)
print(f"Timeout: {t}", flush=True)
print(f"Errors:  {e}", flush=True)
print(f"Time:    {total_time}s", flush=True)

with open(RESULTS_FILE, "w") as fh:
    json.dump(results, fh, indent=2)
print(f"Results saved to {RESULTS_FILE}", flush=True)
