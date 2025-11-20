# Study A: Ready-to-Copy Commands

Notes (matches current script behavior)
- Similarity S uses normalized cosine on the unit circle: `S[i,j] = (1 + cos(Δθ)) / 2`
- Gold generator samples k labels using normalized cosine weights around a random latent angle (no exponentials, no hard cutoffs).
- Deprecated flags removed: no `--tau`, no `--generator-radius`.

## Actual commands

### A1

Alpha sweep (S_alpha = alpha S + (1-alpha)U) - focused on noise effects
```bash
python scripts/semantic_f1/study_a1.py \
  --num-labels 24 \
  --num-examples 1000 \
  --k 3 \
  --p 0.0 0.5 1.0 \
  --near-radii 3 4 \
  --far-radii 5 6 \
  --alpha 0.8 0.6 0.4 0.2 \
  --seed 123 \
  --noise-mode uniform \
  --bootstrap 25 \
  --outdir logs/analysis/semantic_f1/study_a1/alpha_sweep
```

Comprehensive hyperparameter sweep (radius × p × k)
```bash
python scripts/semantic_f1/study_a1.py \
  --num-labels 24 \
  --num-examples 1000 \
  --k 1 2 3 4 \
  --p 0.0 0.2 0.4 0.6 0.8 1.0 \
  --near-radii 1 2 3 4 \
  --far-radii 5 6 7 8 \
  --alpha 1.0 \
  --seed 123 \
  --noise-mode identity \
  --bootstrap 25 \
  --outdir logs/analysis/semantic_f1/study_a1/comprehensive_sweep
```

```bash
python scripts/semantic_f1/study_a1_extended.py \
  --num-labels 24 \
  --num-examples 1500 \
  --k 1 2 3 4 \
  --p 0 0.2 0.4 0.6 0.8 1 \
  --near-radii 1 2 3 4 \
  --far-radii 5 6 7 8 \
  --alpha 0.2 0.4 0.6 0.8 \
  --seed 123 \
  --noise-mode gaussian \
  --bootstrap 25 \
  --outdir logs/analysis/semantic_f1/extended_a1_fast/
```

### A2


```bash
python scripts/semantic_f1/study_a2.py \
  --num-labels 24 \
  --num-examples 100 \
  --k 2 3 \
  --q 0.25 0.5 0.75 \
  --prototype-sizes 2 \
  --near-radii 3 \
  --near-p 1 \
  --imbalance-ratios 0.25 0.5 0.75 \
  --kappa 2.0 \
  --beta 2.0 \
  --seed 123 \
  --bootstrap 15 \
  --outdir logs/analysis/semantic_f1/smoke_a2/quick_test
```

## Camera-Ready (Paper)

```bash
python scripts/semantic_f1/study_a1.py \
  --num-labels 24 \
  --num-examples 20000 \
  --k 1 2 3 \
  --p 0.0 0.25 0.5 0.75 1.0 \
  --near-radii 1 2 \
  --far-radii 3 4 \
  --alpha 1.0 0.75 0.5 0.25 0.0 \
  --seed 123 \
  --noise-mode identity \
  --bootstrap 200 \
  --outdir logs/analysis/semantic_f1/study_a1
```

### Study A2 — Camera-Ready (Paper)

```bash
python scripts/semantic_f1/study_a2.py \
  --num-labels 24 \
  --num-examples 20000 \
  --k 1 2 3 \
  --p 0.0 0.25 0.5 0.75 1.0 \
  --prototype-sizes 2 3 4 \
  --imbalance-ratios 0.3 0.5 0.7 \
  --kappa 2.0 \
  --beta 2.0 \
  --seed 123 \
  --bootstrap 200 \
  --outdir logs/semantic_f1/study_a2
```

## Quick Smoke Tests A1 (Focused)

Alpha sweep (S_alpha = alpha S + (1-alpha)U) - focused on noise effects
```bash
python scripts/semantic_f1/study_a1.py \
  --num-labels 24 \
  --num-examples 1000 \
  --k 3 \
  --p 0.5 \
  --near-radii 3 \
  --far-radii 6 \
  --alpha 1.0 0.8 0.6 0.4 0.2 0.0 \
  --seed 123 \
  --noise-mode uniform \
  --bootstrap 25 \
  --outdir logs/analysis/semantic_f1/a1/alpha_sweep
```

Comprehensive hyperparameter sweep (radius × p × k)
```bash
python scripts/semantic_f1/study_a1.py \
  --num-labels 24 \
  --num-examples 1500 \
  --k 1 2 3 4 \
  --p 0.0 0.2 0.4 0.6 0.8 1.0 \
  --near-radii 1 2 3 4 \
  --far-radii 5 6 7 8 \
  --alpha 1.0 \
  --seed 123 \
  --noise-mode identity \
  --bootstrap 25 \
  --outdir logs/analysis/semantic_f1/smoke_a1/comprehensive_sweep
```


### Study A2 — Quick Smoke Test

```bash
python scripts/semantic_f1/study_a2.py \
  --num-labels 24 \
  --num-examples 100 \
  --k 2 3 \
  --q 0.25 0.5 0.75 \
  --prototype-sizes 2 \
  --near-radii 3 \
  --near-p 1 \
  --imbalance-ratios 0.25 0.5 0.75 \
  --kappa 2.0 \
  --beta 2.0 \
  --seed 123 \
  --bootstrap 15 \
  --outdir logs/analysis/semantic_f1/smoke_a2/quick_test
```

