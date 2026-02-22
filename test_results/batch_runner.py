import subprocess
import os
import sys
import time
import json

PYTHON = os.path.expanduser("~/miniforge3/envs/mp/bin/python")
os.environ["MPLBACKEND"] = "Agg"

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

if __name__ == "__main__":
    batch_id = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 180
    files = sys.argv[3:]

    print(f"=== BATCH {batch_id}: {len(files)} files (timeout={timeout}s) ===", flush=True)
    results = []
    for fp in files:
        results.append(run_file(fp, timeout=timeout))

    p = sum(1 for r in results if r["status"] == "PASS")
    f = sum(1 for r in results if r["status"] == "FAIL")
    t = sum(1 for r in results if r["status"] == "TIMEOUT")
    e = sum(1 for r in results if r["status"] == "ERROR")
    total_time = round(sum(r["time"] for r in results), 1)

    print(f"\n--- BATCH {batch_id} SUMMARY ---", flush=True)
    print(f"Total: {len(results)} | Pass: {p} | Fail: {f} | Timeout: {t} | Error: {e} | Time: {total_time}s", flush=True)

    # Save results
    outfile = f"/mnt/c/Users/simon/Downloads/meep-master/batch_{batch_id}_results.json"
    with open(outfile, "w") as fh:
        json.dump({"batch": batch_id, "results": results, "summary": {"total": len(results), "passed": p, "failed": f, "timeout": t, "errors": e, "total_time": total_time}}, fh, indent=2)
    print(f"Results saved to {outfile}", flush=True)
