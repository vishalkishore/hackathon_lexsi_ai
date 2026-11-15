
#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${1:-CohereLabs/aya-expanse-8b}"
NAME="${2:-baseline}"
GPU="${3:-0}"

OUT="results/${NAME}/capability"
TASKS="${TASKS:-gsm8k mmlu arc_challenge hellaswag truthfulqa_mc2}"
LIMIT="${LIMIT:-}"
# Set GSM8K_SHOTS=0 for zero-shot GSM8K evaluation.
GSM8K_SHOTS="${GSM8K_SHOTS:-5}"

mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"

failed=0

for task in $TASKS; do

    case "$task" in
        gsm8k)
            shots="$GSM8K_SHOTS"
            batch=32
            ;;
        mmlu)
            shots=5
            batch=8
            ;;
        arc_challenge)
            shots=25
            batch=4
            ;;
        hellaswag)
            shots=10
            batch=8
            ;;
        truthfulqa_mc2)
            shots=0
            batch=8
            ;;
        *)
            echo "Unknown task: $task"
            failed=1
            continue
            ;;
    esac

    echo "================================="
    echo "Task: $task"
    echo "Model: $MODEL"
    echo "GPU: $GPU"
    echo "Few-shot: $shots"
    echo "Batch size: $batch"
    echo "================================="

    if MODEL_ID="$MODEL" \
       TASK="$task" \
       SHOTS="$shots" \
       BATCH="$batch" \
       OUT_FILE="$OUT/$task.json" \
       LIMIT="$LIMIT" \
       uv run python -c '
import os
import auditkit as ak

model = os.environ["MODEL_ID"]
if not model.startswith("hf:"):
    model = "hf:" + model

task = os.environ["TASK"]
shots = int(os.environ["SHOTS"])
batch = int(os.environ["BATCH"])
output = os.environ["OUT_FILE"]
limit = os.environ["LIMIT"]

kwargs = {}

if task == "gsm8k":
    kwargs["gen_kwargs"] = (
        "temperature=0,max_gen_toks=256"
    )

result = ak.run_lmeval(
    tasks=task,
    model=model,
    config=ak.RunConfig(
        batch_size=batch,
        num_fewshot=shots,
        limit=int(limit) if limit else None,
    ),
    dtype="bfloat16",
    apply_chat_template=True,
    **kwargs,
)

print(result.summary())
print(result.headline)

result.save(output)
print("Saved:", output)
' 2>&1 | tee "$OUT/$task.log"; then

        echo "$task: SUCCESS"
    else
        echo "$task: FAILED" | tee -a "$OUT/failed.txt"
        failed=1
    fi
done

if (( failed )); then
    echo "Some evaluations failed."
    exit 1
fi

echo "All task evaluations completed."
echo "Results: $OUT"
