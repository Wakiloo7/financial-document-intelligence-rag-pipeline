# LLaMA Factory LoRA Fine-tuning Commands

This experiment demonstrates a small supervised fine-tuning workflow using LLaMA Factory and LoRA on financial RAG-style instruction data.

## 1. Create and activate environment

py -3.11 -m venv .venv-llama
.\.venv-llama\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r experiments\llama_factory_lora_finetuning\requirements-llamafactory.txt

## 2. Run LoRA SFT training

llamafactory-cli train experiments\llama_factory_lora_finetuning\configs\tinyllama_lora_sft.yaml

## 3. Optional inference after training

llamafactory-cli chat experiments\llama_factory_lora_finetuning\configs\tinyllama_lora_sft.yaml

## Notes

- This is a compact portfolio experiment, not a production model.
- The goal is to demonstrate instruction dataset preparation, LoRA configuration, LLaMA Factory usage, and evaluation workflow.
- For GPU training, CUDA-enabled PyTorch is recommended.
- On CPU, training may be slow.
