python scripts/training/training_lora_llm.py SemEval --gridparse-config ./configs/SemEval/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric jaccard_score --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test

python scripts/training/training_lora_llm.py GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric jaccard_score --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test

python scripts/training/training_lora_llm.py GoEmotions --gridparse-config ./configs/GoEmotions/config_full_emotions.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric jaccard_score --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test

python scripts/training/training_lora_llm.py MFRC --gridparse-config ./configs/MFRC/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric jaccard_score --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test