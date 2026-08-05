# Statistical significance (clip-level, paired)

McNemar (continuity-corrected) on pooled paired clips; bootstrap 95% CI on clip-level dAcc (5000 resamples). CLIP-level — not subject-level.

| Comparison (A vs B) | acc A | acc B | dAcc (A-B) | 95% CI | McNemar chi2 | p |
|---|---|---|---|---|---|---|
| PoseGated multi-[0,1] vs RGB baseline | 94.8 | 90.6 | **+4.1** | [+3.8, +4.4] | 763.6 | 4.44e-168 |
| PoseGated multi-[0,1] vs Early concat | 94.8 | 93.8 | **+1.0** | [+0.7, +1.2] | 58.6 | 1.92e-14 |
| PoseGated multi-[0,1] vs SE [0] | 94.8 | 92.1 | **+2.7** | [+2.4, +3.0] | 358.8 | 5.22e-80 |
| PoseGated multi-[0,1] vs Cross-attn [4] | 94.8 | 90.2 | **+4.5** | [+4.2, +4.8] | 817.8 | 7.41e-180 |
| PoseGated multi-[0,1] vs PoseGated full [0-4] | 94.8 | 91.0 | **+3.8** | [+3.5, +4.0] | 762.7 | 7.01e-168 |

**bold** dAcc = 95% CI excludes 0.

Per-fold paired accuracies (A / B):

- PoseGated multi-[0,1] vs RGB baseline: f0: 97.1/93.4  f1: 94.9/90.1  f2: 92.5/88.8
- PoseGated multi-[0,1] vs Early concat: f0: 97.1/95.7  f1: 94.9/95.7  f2: 92.5/89.8
- PoseGated multi-[0,1] vs SE [0]: f0: 97.1/93.2  f1: 94.9/94.1  f2: 92.5/88.7
- PoseGated multi-[0,1] vs Cross-attn [4]: f0: 97.1/87.8  f1: 94.9/94.5  f2: 92.5/87.4
- PoseGated multi-[0,1] vs PoseGated full [0-4]: f0: 97.1/92.7  f1: 94.9/93.4  f2: 92.5/86.6
