The environment is configured in CBTSQ-CR/env.ymal

Our experiments include dataset generation, model dialogue, and model fine-tuning. The prompts for both dataset generation and model dialogue are provided in the paper.

The model fine-tuning part refers to https://github.com/XplainMind/LLMindCraft.

Fine-tune the model using CBTSQ-CR/src/train/scripts/run_sft.sh to obtain checkpoints, then merge the checkpoints using CBTSQ-CR/src/train/scripts/merge_lora.sh.



The dataset and all the dialogue data will be made public later.
