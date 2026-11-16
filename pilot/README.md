# Pilot: uncertainty-aware weak-to-strong on CIFAR-10

Small end-to-end test of the project pipeline, run before the proposal (Nov 15, NYCU cluster, one RTX 5050).

## What the script does (`pilot.py`, one seed per run)
1. Splits the CIFAR-10 training set (stratified, fixed by the seed): 20,000 teacher / 25,000 student / 5,000 validation. The 10,000 test images are kept for evaluation only.
2. Trains the weak teacher (small AlexNet-style CNN, 3 epochs, true labels) on the teacher set.
3. The teacher labels the student, validation and test sets. These are the weak labels.
4. Extracts frozen DINO ViT-S/16 features once, and trains a small MLP head (384 → 256 → 10, dropout 0.2) on the weak labels with each strategy below. Early stopping uses weak-label validation accuracy, so no true label is used in training.
5. Evaluates every head on the test set with true labels.

| Name | Training strategy |
|---|---|
| A | cross-entropy on weak labels only (α = 0) |
| B | A02 confidence loss, fixed α = 0.5, hard self-target |
| B′ (`Bprime`) | B with α equal to the mean α used by C |
| C | α per image: α_i = α_max (1 − u_i), u_i = normalised MC-Dropout entropy |
| Cnorm | C rescaled so the batch mean of α_i is α_max |
| D25 / D50 / D75 | fixed α; hard self-target for the fraction q of each batch with the lowest model uncertainty (MC-Dropout mutual information), soft target (mean prediction) for the rest |
| Drand50 | D50 with a random gate (control) |
| ceiling | same head trained on true labels |

## How to run
```
python pilot.py --smoke                 # synthetic data, CPU, a few seconds
python pilot.py --prepare               # login node: download CIFAR-10 and DINO
sbatch run_pilot.sbatch --seed 0        # one seed, a few minutes
```
Raw outputs are in `results/results_seed{0,1,2}_T3.json`.

## Results: Pilot with method D (Nov 15, seeds 0/1/2, jobs 788166, 788229, 788279)

Teacher accuracy: 0.627 / 0.598 / 0.629. Mean ± std over 3 seeds, CIFAR-10 test set.
D = fixed α = 0.5, hard self-target for the fraction q of each batch with the lowest model uncertainty (MC-Dropout mutual information), soft target for the rest. Drand50 = same with a random gate.

| Model | Accuracy | PGR | ECE | Acc@80 % cov. | AUROC errors | AUROC copied errors |
|---|---|---|---|---|---|---|
| A (naive) | 0.715 ± 0.024 | 0.29 | 0.033 | 0.792 | 0.795 | 0.740 |
| B (A02, α = 0.5) | 0.725 ± 0.044 | 0.32 | 0.090 | 0.802 | 0.797 | 0.737 |
| B′ (α = 0.38) | 0.731 ± 0.027 | 0.33 | 0.049 | 0.809 | 0.800 | 0.743 |
| C (α per image) | 0.728 ± 0.017 | 0.33 | 0.051 | 0.807 | 0.805 | 0.746 |
| Cnorm | 0.730 ± 0.020 | 0.33 | 0.071 | 0.810 | 0.807 | 0.749 |
| D, q = 0.25 | 0.725 ± 0.037 | 0.32 | 0.041 | 0.801 | 0.800 | 0.745 |
| D, q = 0.50 | 0.728 ± 0.022 | 0.32 | 0.033 | 0.807 | 0.806 | 0.749 |
| D, q = 0.75 | 0.723 ± 0.004 | 0.31 | 0.068 | 0.803 | 0.806 | 0.748 |
| Drand50 (random gate) | 0.735 ± 0.036 | 0.35 | 0.043 | 0.813 | 0.804 | 0.749 |
| Ceiling | 0.956 ± 0.001 | 1 | 0.024 | 0.996 | 0.944 | 0.928 |

Coverage and AUROC use MC-Dropout entropy (20 passes).

### Gate check (H3): share of wrong teacher labels, hardened / soft examples
| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D25 | 0.05 / 0.48 | 0.05 / 0.51 | 0.06 / 0.49 |
| D50 | 0.15 / 0.59 | 0.16 / 0.63 | 0.17 / 0.60 |
| D75 | 0.27 / 0.68 | 0.29 / 0.73 | 0.28 / 0.69 |
| Drand50 | 0.37 / 0.37 | 0.40 / 0.40 | 0.38 / 0.38 |

### Paired differences per seed (seed 0, 1, 2 → mean)
| | Accuracy | AUROC errors | AUROC copied | ECE |
|---|---|---|---|---|
| D50 − B | −0.018, +0.034, −0.009 → +0.002 | +0.005, +0.019, +0.004 → +0.010 | +0.005, +0.029, +0.004 → +0.013 | −0.020, −0.105, −0.044 → −0.056 |
| C − B | −0.024, +0.036, −0.003 → +0.003 | +0.004, +0.016, +0.004 → +0.008 | +0.002, +0.023, +0.004 → +0.010 | +0.006, −0.091, −0.031 → −0.039 |
| D50 − Drand50 | −0.024, +0.008, −0.007 → −0.008 | −0.000, +0.002, +0.005 → +0.002 | −0.007, +0.003, +0.005 → +0.000 | +0.002, −0.031, −0.001 → −0.010 |

### Reading
- The gate does what it should: hardened examples carry far fewer teacher errors than soft ones.
- Accuracy: no method is separated from B; seed spread (0.02 to 0.04) is larger than every gap.
- D50 vs B: lower ECE and slightly higher error-detection AUROC on all 3 seeds, but small.
- D50 is not better than the random gate. The benefit over B seems to come from soft targets on half the batch, not from which examples are chosen.
- B is unstable (ECE 0.134 on seed 1), and most gains over B come from that seed.
