# TODO: KingaMetric Optuna Model Enhancements (Approved Plan - Target AUC >=0.85)

## Steps (to be checked off as completed):
- [ ] Step 1: Create TODO.md with this breakdown ✓
- [x] Step 2: Check `improved_credit_risk.csv` and decide dataset (prefer if better baseline AUC). **Done: Updated notebook to load improved dataset (36 cols, same shape/rate, 37 engineered feats like normalized_dti, Borrower_Tier). User ran, CV AUC=0.646 <0.85.**
- [ ] Step 3: Add data split (train/val/test), enhanced preprocessing (outliers, eng from SQL).
- [x] Step 4: Improve feature selection (e.g., recursive or permutation importance). **User manual: top-30 features, AUC=0.645 still low. Keeping all features next for full power.**
- [x] Step 5: Edit Optuna: 100+ trials, early stopping/pruning, sampler improvements. **Updated to 100 trials with MedianPruner. User ran, AUC=0.6468 still <0.85 (top-20 features too restrictive, low trials effect minimal).**
- [ ] Step 6: Add SHAP explainability, baselines, advanced metrics.
- [ ] Step 7: Persist model (`joblib`), predictions CSV, plots.
- [ ] Step 8: Test run notebook for AUC >=0.85, update TODO.
- [ ] Step 9: attempt_completion once validated.

