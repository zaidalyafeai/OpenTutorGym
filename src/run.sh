TUTOR_MODEL="google/gemini-flash-1.5"
STUDENT_MODEL="Qwen/Qwen2.5-7B-Instruct"
# STUDENT_MODEL="gemma3:12b"
JUDGE_MODEL="google/gemini-flash-1.5"
# JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="gsm8k"
MODE="tutor:0"
MAX_COMPLETION_TOKENS=8192
NUM_WORKERS=64
MAX_DATASET_SIZE=128
DIRECTORY="results"

python main.py --dataset ${DATASET} --student_model ${STUDENT_MODEL} --tutor_model ${TUTOR_MODEL} \
      --max_completion_tokens ${MAX_COMPLETION_TOKENS} \
      --num_workers ${NUM_WORKERS} \
      --max_dataset_size ${MAX_DATASET_SIZE} \
      --directory ${DIRECTORY} \
      --mode ${MODE} \
      --use_judge \
      --judge_model ${JUDGE_MODEL} \
      --overwrite