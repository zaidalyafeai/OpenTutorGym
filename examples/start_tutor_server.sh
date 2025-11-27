source /ibex/ai/home/alyafez/.vllm/bin/activate
CUDA_VISIBLE_DEVICES=0 vllm serve \
        /ibex/ai/home/alyafez/models/$1 \
        --host 0.0.0.0 \
        --port $2 \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --max-model-len 4096 \
        --max-num-seqs 50 \
        --dtype=bfloat16 \
	--served-model-name $1
