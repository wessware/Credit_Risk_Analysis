# TODO: Improve Default_Flag generation for balanced classes
## Steps:
1. **[PENDING]** Add diagnostic cells after target code: compute current Default_Flag.value_counts(normalize=True), Borrower_Tier.value_counts(normalize=True), groupby mean.
2. **[PENDING]** Implement Variant 1: Tune weights/beta/bins, rerun stats.
3. **[PENDING]** Implement Variant 2: qcut tiers + tier multipliers, rerun.
4. **[PENDING]** Variant 3: Heavier beta bias, check overall/subprime rates.
5. **[PENDING]** Fine-tune to targets (subprime 17-24%, repayment 70-75%), select best.
6. **[PENDING]** Save final df to datasets/improved_credit_risk.csv.
7. **[PENDING]** Verify final stats, attempt_completion.

Track progress by updating status as [DONE] after each.

