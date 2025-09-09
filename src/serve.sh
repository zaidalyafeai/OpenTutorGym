source /ibex/ai/home/alyafez/.vllm/bin/activate
vllm serve \
        /ibex/ai/home/alyafez/models/$1 \
        --host localhost \
        --port $3 \
        --gpu-memory-utilization $2 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --max-model-len 2084 \
        --max-num-seqs 50 \
        --dtype=bfloat16 \
	--served-model-name $1
