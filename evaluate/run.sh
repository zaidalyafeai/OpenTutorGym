TUTOR_MODEL="Qwen/Qwen2.5-7B-Instruct"
STUDENT_MODEL="Qwen/Qwen2.5-3B-Instruct"
# STUDENT_MODEL="gemma3:12b"
JUDGE_MODEL="deepseek/deepseek-chat-v3-0324:free"
DATASET="gsm8k"
MODE="tutor:3"
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
      --overwrite