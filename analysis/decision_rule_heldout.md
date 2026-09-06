# Validation-tuned decision rules (heldout_pose_single_L3; clip level; tuned on inner VAL, applied to held-out TEST)

Rules: (a) 3-class argmax(p*w), w tuned on val for balanced accuracy; (b) binary P(ASD)>=t, t tuned on val.
Default = plain argmax (paper). All numbers %.

| fold | w (ASD,DHS,LCS) | t | 3-cls bacc def→tuned | macro-F1 def→tuned | acc def→tuned | recall ASD/DHS/LCS def | recall ASD/DHS/LCS tuned | binary sens/spec/bacc def | binary sens/spec/bacc tuned |
|---|---|---|---|---|---|---|---|---|---|
| 0 | [1.0, 0.1, 0.1] | 0.961 | 51.4→48.5 | 50.6→47.0 | 71.4→69.3 | 97/57/0 | 100/46/0 | 97/40/68.4 | 84/46/65.3 |
| 1 | [1.0, 100.0, 19.95] | 0.992 | 54.2→55.8 | 52.7→51.0 | 74.6→70.4 | 99/64/0 | 76/92/0 | 99/45/71.7 | 73/81/76.9 |
| 2 | [1.0, 0.16, 0.1] | 0.147 | 44.3→42.7 | 43.0→40.5 | 61.7→61.6 | 86/47/0 | 91/37/0 | 86/40/62.8 | 90/26/58.2 |

**Mean ± std over folds (TEST):**

| metric | default | val-tuned |
|---|---|---|
| 3-class balanced acc | 50.0 ± 4.2 | 49.0 ± 5.4 |
| 3-class macro-F1 | 48.8 ± 4.2 | 46.2 ± 4.3 |
| 3-class accuracy | 69.2 ± 5.5 | 67.1 ± 3.9 |
| recall ASD | 93.8 ± 5.8 | 88.8 ± 9.8 |
| recall DHS | 56.2 ± 6.9 | 58.3 ± 23.8 |
| recall LCS-HipOA | 0.0 ± 0.0 | 0.0 ± 0.0 |
| binary sensitivity | 93.8 ± 5.8 | 82.5 ± 7.0 |
| binary specificity | 41.4 ± 2.4 | 51.0 ± 22.4 |
| binary balanced acc | 67.6 ± 3.7 | 66.8 ± 7.7 |
