source /ibex/ai/home/alyafez/.vllm/bin/activate
vllm serve \
        /ibex/ai/home/alyafez/models/gemma-3-12b-it \
        --host localhost \
        --port 8787 \
        --gpu-memory-utilization 0.95 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --max-model-len 8192 \
        --max-num-seqs 50 \
        --dtype=bfloat16 \
	--served-model-name gemma-3-12b-it
