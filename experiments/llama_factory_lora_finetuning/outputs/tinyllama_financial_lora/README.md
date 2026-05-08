---
library_name: peft
license: other
base_model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
tags:
- base_model:adapter:TinyLlama/TinyLlama-1.1B-Chat-v1.0
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: tinyllama_financial_lora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# tinyllama_financial_lora

This model is a fine-tuned version of [TinyLlama/TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on the financial_rag_instruction_sample dataset.
It achieves the following results on the evaluation set:
- Loss: 2.6567

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- training_steps: 3

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.9237        | 0.3333 | 2    | 2.6578          |


### Framework versions

- PEFT 0.17.1
- Transformers 4.57.1
- Pytorch 2.11.0+cpu
- Datasets 4.0.0
- Tokenizers 0.22.2