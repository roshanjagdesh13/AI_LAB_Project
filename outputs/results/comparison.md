# Model comparison

All four models were scored on the same held-out test set (3,781 events, never used for fitting, tuning or early stopping). A prediction is 'gamma' when the model's gamma probability is at least 0.5.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8736 | 0.8883 | 0.9221 | 0.9049 | 0.9327 |
| Neural Network | 0.8754 | 0.8901 | 0.923 | 0.9062 | 0.9289 |
| SVM (RBF) | 0.8693 | 0.9008 | 0.8986 | 0.8997 | 0.9283 |
| Logistic Regression | 0.7792 | 0.8402 | 0.8167 | 0.8283 | 0.8349 |

## Winner: Random Forest

**Random Forest** is selected because it has the highest ROC-AUC (0.9327). ROC-AUC measures how well a model ranks gammas above hadrons across *all* thresholds, so it is not distorted by the class imbalance or by the 0.5 cut-off. F1 is the tie-breaker.

- Runner-up: Neural Network (AUC 0.9289); Random Forest leads by 0.0038 AUC and -0.0013 F1.
- The gap is small (< 0.005 AUC); on a test set of this size the two should be considered close.
- Versus the linear baseline (Logistic Regression, AUC 0.8349), Random Forest improves AUC by +0.0977 and accuracy by +0.0944, i.e. the classes are not well separated by a straight boundary.
- Errors of Random Forest on the test set: 286 hadrons mislabelled as gamma (false positives) and 192 gammas missed (false negatives). Recall is 0.922 and precision 0.888.
- The Random Forest relies most on `fAlpha` (26.5% of importance), then `fLength` (14.2%).
- Training time on CPU (including tuning): Logistic Regression 0s, SVM (RBF) 62s, Random Forest 57s, Neural Network 173s.

## Tuned settings

- **SVM (RBF):** C=10, gamma=scale, 5-fold CV ROC-AUC 0.9222
- **Random Forest:** max_depth=None, min_samples_leaf=1, n_estimators=200, 5-fold CV ROC-AUC 0.9316
- **Neural Network:** num_nodes=32, dropout=0.2, lr=0.005, batch_size=32, validation loss 0.3227, 27 epochs
