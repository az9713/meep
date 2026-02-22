#!/bin/bash
# Run all meep examples and tests in 5 parallel batches
PYTHON=~/miniforge3/envs/mp/bin/python
RUNNER=/mnt/c/Users/simon/Downloads/meep-master/batch_runner.py
export MPLBACKEND=Agg

EXAMPLES_DIR=/mnt/c/Users/simon/Downloads/meep-master/python/examples
TESTS_DIR=/mnt/c/Users/simon/Downloads/meep-master/python/tests

# Get sorted file lists
mapfile -t EXAMPLES < <(ls "$EXAMPLES_DIR"/*.py | sort)
mapfile -t TESTS < <(ls "$TESTS_DIR"/test_*.py | sort)

ALL=("${EXAMPLES[@]}" "${TESTS[@]}")
TOTAL=${#ALL[@]}
echo "Total files: $TOTAL"

# Split into 5 batches
BATCH_SIZE=$(( (TOTAL + 4) / 5 ))

for i in 0 1 2 3 4; do
    START=$(( i * BATCH_SIZE ))
    BATCH=("${ALL[@]:$START:$BATCH_SIZE}")
    if [ ${#BATCH[@]} -gt 0 ]; then
        echo "Starting batch $i with ${#BATCH[@]} files..."
        $PYTHON "$RUNNER" "$i" 180 "${BATCH[@]}" > /mnt/c/Users/simon/Downloads/meep-master/batch_${i}_output.txt 2>&1 &
    fi
done

echo "All batches launched. Waiting for completion..."
wait
echo "All batches complete!"

# Merge results
$PYTHON -c "
import json, glob, os
batches = sorted(glob.glob('/mnt/c/Users/simon/Downloads/meep-master/batch_*_results.json'))
all_results = []
for bf in batches:
    with open(bf) as f:
        data = json.load(f)
        all_results.extend(data['results'])

p = sum(1 for r in all_results if r['status'] == 'PASS')
f = sum(1 for r in all_results if r['status'] == 'FAIL')
t = sum(1 for r in all_results if r['status'] == 'TIMEOUT')
e = sum(1 for r in all_results if r['status'] == 'ERROR')
total_time = round(sum(r['time'] for r in all_results), 1)

combined = {
    'total': len(all_results),
    'passed': p, 'failed': f, 'timeout': t, 'errors': e,
    'total_time': total_time,
    'results': all_results
}
outfile = '/mnt/c/Users/simon/Downloads/meep-master/test_results.json'
with open(outfile, 'w') as fh:
    json.dump(combined, fh, indent=2)

print()
print('=' * 60)
print('COMBINED RESULTS')
print('=' * 60)
print(f'Total:   {len(all_results)}')
print(f'Passed:  {p}')
print(f'Failed:  {f}')
print(f'Timeout: {t}')
print(f'Errors:  {e}')
print(f'Time:    {total_time}s')
print(f'Results saved to {outfile}')

# Print failures
if f > 0:
    print()
    print('FAILED:')
    for r in all_results:
        if r['status'] == 'FAIL':
            print(f\"  {r['name']}: {r['error'][:100]}\")

if t > 0:
    print()
    print('TIMEOUTS:')
    for r in all_results:
        if r['status'] == 'TIMEOUT':
            print(f\"  {r['name']}\")

if e > 0:
    print()
    print('ERRORS:')
    for r in all_results:
        if r['status'] == 'ERROR':
            print(f\"  {r['name']}: {r['error'][:100]}\")
"
