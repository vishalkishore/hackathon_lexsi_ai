
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${1:-CohereLabs/aya-expanse-8b}"
NAME="${2:-baseline}"
GPU="${3:-1}"

OUT="results/${NAME}/safety"
mkdir -p "$OUT"

export CUDA_VISIBLE_DEVICES="$GPU"

# Override with TASKS="advbench xstest" if needed.
TASKS="${TASKS:-advbench wildjailbreak orbench_hard xstest harmbench}"
read -r -a BENCHMARKS <<< "$TASKS"

printf 'Model: %s\nGPU: %s\n' "$MODEL" "$GPU" \
  | tee "$OUT/run_info.txt"

failed=0

for task in "${BENCHMARKS[@]}"; do
    echo "=================================="
    echo "Running: $task"
    echo "=================================="

    mkdir -p "$OUT/$task"

    if uv run safetune eval \
        --model "$MODEL" \
        --dataset "$task" \
        --eval-backend vllm \
        --output "$OUT/$task" \
        2>&1 | tee "$OUT/$task.log"; then

        echo "$task: SUCCESS"
    else
        echo "$task: FAILED" | tee -a "$OUT/failed.txt"
        failed=1
    fi
done

if (( failed )); then
    echo "Some benchmarks failed. Check logs in $OUT"
    exit 1
fi

echo "All benchmarks completed."
echo "Results directory: $OUT"
