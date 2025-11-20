# a1

python scripts/semantic_f1/study_a1.py  \
    --num-labels 24 --num-examples 1000 \
    --k 1 2 3 4 --p 0.0 0.2 0.4 0.6 0.8 1.0 \
    --near-radii 1 2 3 4 --far-radii 5 6 7 8 \
    --alpha 1.0 --seed 123 --noise-mode identity \
    --bootstrap 25 --outdir logs/analysis/semantic_f1/study_a1/comprehensive_sweep

python scripts/semantic_f1/study_a1.py \
    --num-labels 24 --num-examples 1000 \
    --k 3 --p 0.0 0.5 1.0 --near-radii 3 4 --far-radii 5 6 \
    --alpha 0.8 0.6 0.4 0.2 --seed 123 --noise-mode gaussian --bootstrap 25 \
    --outdir logs/analysis/semantic_f1/study_a1/alpha_sweep_gaussian

# a2

python scripts/semantic_f1/study_a2.py \
    --num-labels 24 --num-examples 1000 \
    --k 1 2 3 4 --q 0 0.2 0.4 0.6 0.8 1 --prototype-sizes 2 \
    --near-radii 3 --near-p 1 --imbalance-ratios 0.25 0.5 0.75 \
    --kappa 2.0 --beta 2.0 --seed 123 --bootstrap 25 \
    --outdir logs/analysis/semantic_f1/study_a2/

# a3

python scripts/semantic_f1/study_a3.py \
    --geometry union --union-components ringA=ring:24 ringB=ring:24  --num-labels 48 \
    --num-examples 1000 --k 1 2 3 4 --p 1.0 --near-radii 1 2 3 4 --far-radii 5 6 7 8 \
    --alpha 1.0 0.5  --union-cross-jump-prob 0 0.2 0.4 0.6 0.8 1 --union-ring-min-sim 0.35 \
    --deceptive-ring-gap 0.2 --noise-mode uniform --bootstrap 25 --seed 123 \
    --outdir logs/analysis/semantic_f1/study_a3_new

# a4

python scripts/semantic_f1/study_a4.py hungarian \
    --num-label 192 --num-examples 1000 --k 2 \
    --outdir logs/analysis/semantic_f1/study_a4_hungarian --rad 2 --temp 0.05 \
    --no-boot --prediction-counts 1 3 5 7 9 11 13 15 17 --pred-temp 0.1

python scripts/semantic_f1/study_a4.py recall \
    --num-label 96 --num-examples 1000 --k 6 8 10 --p 1 \
    --outdir logs/analysis/semantic_f1/study_a4_recall \
    --rad 2 --temp 0.05 --no-boot

python scripts/semantic_f1/study_a4.py precision \
    --num-label 96 --num-examples 1000 --k 6 8 10 --p 1 \
    --outdir logs/analysis/semantic_f1/study_a4_precision \
    --rad 2 --temp 0.05 --no-boot