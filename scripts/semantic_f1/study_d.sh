#!/bin/bash

# goemotions

python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct \
    --shot 2 --annotation-mode aggregate --quantization fp8 --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 2 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 2 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct \
    --shot 0 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0

python scripts/prompting/vllm_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 0 --annotation-mode aggregate --quantization bitsandbytes \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0

# openai model

python scripts/prompting/api_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 0 --annotation-mode aggregate \
    --alternative {model_name}-inference-{shot}-shot --seed 0

# semeval

python scripts/prompting/vllm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct \
    --shot 2 --annotation-mode aggregate --quantization fp8 --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 2 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train --instruction configs/PersuasionForGood/instruction-semeval.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct \
    --shot 0 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-semeval-{shot}-shot --seed 0

python scripts/prompting/vllm_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train --instruction configs/PersuasionForGood/instruction-semeval.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 0 --annotation-mode aggregate --quantization bitsandbytes \
    --alternative {model_name_or_path}-inference-semeval-{shot}-shot --seed 0

# openai model

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 2 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    PersuasionForGood --gridparse-config ./configs/PersuasionForGood/config.yaml --test-split train --instruction configs/PersuasionForGood/instruction-semeval.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 0 --annotation-mode aggregate \
    --alternative {model_name}-inference-{shot}-shot --seed 0

