# OpenTutorGym

OpenTutorGym is an open framework for training and evaluating math tutors. It supports the following features

- [x] Creating synthetic conversations between a student and a tutor
- [x] Training math tutors using LLMs using supervised finetuning and preference optimization
- [x] Evaluating math tutors using LLM-as-a-Judge

## Installation

This repository uses `uv` as a package manager. You can install it using `curl`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing `uv`, you can initialize the project using `uv init`. The install all the dependencies using `uv install`.

## Backends

OpenTutorGym supports the following backends

### vLLM
To use the vLLM backend, you need to have vLLM installed. First, create an environment `.vllm` and install vLLM. You can install it using `pip install vllm`. Then, you can start the server using `./scripts/start_server.sh <model_name> <port>`. You can download models usng the huggingface-cli 
`huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir Qwen2.5-7B-Instruct`


### OpenRouter
To use the OpenRouter backend, you need to have a valid key from OpenRouter. You can get it from [here](https://openrouter.ai/settings/keys). Then you need to create a file named `.env` with the following key `OPENROUTER_API_KEY=<your_key>`. 

### Transformers
The transformers library is enabled by default when you install the dependencies.

## Examples 
We provide multiple examples in the `examples` directory. You can run them using `uv run <example_name>`

## Data generation

To generate data, you can use the `scripts/generate.slurm` script. It will generate data using the specified model and save it to the specified directory.

```bash
sbatch scripts/generate.slurm
```

## Fine-tuning

To fine-tune a model, you can use the `scripts/fine_tune.slurm` script. It will fine-tune the specified model on the specified dataset and save it to the specified directory.

```bash
sbatch scripts/fine_tune.slurm
```

## Examples

We provide multiple examples in the `examples` directory. You can run them using `uv run <example_name>`
1. `finetune_dpo.py` - Finetunes a model using DPO
2. `finetune_sft.py` - Finetunes a model using SFT
3. `inspect_dataset.py` - Inspects a dataset
4. `evaluate_conversation.py` - Evaluates a conversation using a model
5. `generate_conversation.py` - Generates a conversation using a model
6. `generate_conversation_image.py` - Generates a conversation from a vision dataset

## License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






