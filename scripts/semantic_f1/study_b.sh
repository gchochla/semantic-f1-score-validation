#!/bin/bash

# local model, 70b

python scripts/prompting/vllm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-full-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml --instruction ./configs/MFRC/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    MovieLens --gridparse-config ./configs/MovieLens/config.yaml --instruction ./configs/MovieLens/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    MMLUPro --gridparse-config ./configs/MMLU-Pro/config.yaml --instruction ./configs/MMLU-Pro/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    TREC --gridparse-config ./configs/TREC/config.yaml --instruction ./configs/TREC/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 25 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/vllm_prompting_clsf.py \
    Boxes --gridparse-config ./configs/Boxes/config.yaml --instruction ./configs/Boxes/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-3.3-70B-Instruct \
    --shot 3 --annotation-mode aggregate --quantization bitsandbytes --label-format json \
    --alternative {model_name_or_path}-vllm-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

# local model, 7b

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml --instruction ./configs/SemEval/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-full-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml --instruction ./configs/MFRC/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    MovieLens --gridparse-config ./configs/MovieLens/config.yaml --instruction ./configs/MovieLens/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    MMLUPro --gridparse-config ./configs/MMLU-Pro/config.yaml --instruction ./configs/MMLU-Pro/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct \
    --shot 10 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    TREC --gridparse-config ./configs/TREC/config.yaml --instruction ./configs/TREC/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 25 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

CUDA_VISIBLE_DEVICES=2 python scripts/prompting/vllm_prompting_clsf.py \
    Boxes --gridparse-config ./configs/Boxes/config.yaml --instruction ./configs/Boxes/instruction.txt \
    --model-name-or-path meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-3.1-8B-Instruct --label-format json \
    --shot 3 --annotation-mode aggregate --quantization fp8 \
    --alternative {model_name_or_path}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

# openai model

python scripts/prompting/api_prompting_clsf.py \
    SemEval --gridparse-config ./configs/SemEval/config.yaml --instruction ./configs/SemEval/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml --instruction ./configs/GoEmotions/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    MFRC --gridparse-config ./configs/MFRC/config.yaml --instruction ./configs/MFRC/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    MovieLens --gridparse-config ./configs/MovieLens/config.yaml --instruction ./configs/MovieLens/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    MMLUPro --gridparse-config ./configs/MMLU-Pro/config.yaml --instruction ./configs/MMLU-Pro/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 5 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    TREC --gridparse-config ./configs/TREC/config.yaml --instruction ./configs/TREC/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 25 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300

python scripts/prompting/api_prompting_clsf.py \
    Boxes --gridparse-config ./configs/Boxes/config.yaml --instruction ./configs/Boxes/instruction.txt \
    --model-name gpt-4.1-mini gpt-4o-mini \
    --shot 3 --annotation-mode aggregate --label-format json \
    --alternative {model_name}-inference-{shot}-shot --seed 0 1 2 3 4 --test-debug-len 300