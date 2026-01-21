# alignment-experiments
End-to-end LLM training, alignment, and safety evaluation.

## Overview
This repository contains a series of controlled, from-scratch experiments investigating how modern alignment techniques affect the behavior and internal structure of language models.

Specifically, the experiments examine how instruction tuning and RLHF influence:
* Model behavior and failure modes during generation
* Reward hacking and behavioral regressions introduced by optimization against learned reward models
* Safety-relevant generalization to novel or adversarial prompts
* Changes in internal representations before and after alignment
  
All models are trained either from scratch or with minimal finetuning, using small-scale transformer architectures. This design prioritizes interpretability, mechanistic understanding, and failure analysis over raw benchmark performance.

## Primary Question
How do instruction tuning and RLHF reshape language model behavior, and what new failure modes do they introduce?
