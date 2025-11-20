# # LoRA

python scripts/training/training_lora_llm.py SemEval --gridparse-config ./configs/SemEval/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test --alternative {model_name_or_path}-{early_stopping_metric} --seed 0 1 2 3 4

python scripts/training/training_lora_llm.py SemEval --gridparse-config ./configs/SemEval/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test

python scripts/training/training_lora_llm.py GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test --alternative {model_name_or_path}-{early_stopping_metric} --seed 0 1 2 3 4

python scripts/training/training_lora_llm.py GoEmotions --gridparse-config ./configs/GoEmotions/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test --alternative {model_name_or_path}-{early_stopping_metric} --seed 0 1 2 3 4
    
python scripts/training/training_lora_llm.py MFRC --gridparse-config ./configs/MFRC/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test --alternative {model_name_or_path}-{early_stopping_metric} --seed 0 1 2 3 4

python scripts/training/training_lora_llm.py MFRC --gridparse-config ./configs/MFRC/config.yaml --model-name meta-llama/Llama-3.2-1B-Instruct \
    --num-train-epochs 30 --train-batch-size 32 --max-len 128 --early-stopping-patience 3 --early-stopping-metric samples_f1 --lr 1e-4 \
    --lora-r 4 --dev-split dev --test-split test --alternative {model_name_or_path}-{early_stopping_metric} --seed 0 1 2 3 4

# DEMUX

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py SemEval \
    --gridparse-config ./configs/SemEval/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 30 --max-length 64 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric semantic_samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py GoEmotions \
    --gridparse-config ./configs/GoEmotions/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric samples_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric semantic_micro_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9

python scripts/training/demux.py MFRC \
    --gridparse-config ./configs/MFRC/config.yaml \
    --train-split train --dev-split dev --test-split test \
    --num-train-epoch 20 --max-length 128 --accelerate --warmup 0.1 \
    --model-name bert-base-uncased --text-preprocessor true --seed 0 \
    --train-batch-size 128 --eval-batch-size 256 --early-stopping-patience 3 --early-stopping-metric micro_f1 \
    --intra-loss-coef 0.2 --alternative {model_name_or_path}-{early_stopping_metric}-0.2 --seed 5 6 7 8 9