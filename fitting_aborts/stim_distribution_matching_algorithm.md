# Stim Distribution Matching Against a Reference Truncation

This note explains the matching algorithm used in
`fit_only_LED_off_with_LED_ON_fits_truncate_NOT_censor_ABL_delay_no_choice_match_ref_stim.py`.

The goal is:

- choose a reference truncation, for example `115 ms`
- choose a larger target truncation, for example `130 ms`
- fit the target-truncation data only after subsampling it so that the `intended_fix`
  distribution matches the reference-truncation data

In this script, `intended_fix` is the stimulus-onset variable whose distribution is being
matched.

## Why Matching Is Needed

If the truncation threshold changes, the retained trial set changes.

For example:

- with `RTwrtStim <= 115 ms`, one set of valid trials is retained
- with `RTwrtStim <= 130 ms`, a larger set of valid trials is retained

The 130 ms set is a superset of the 115 ms set, but its `intended_fix` distribution can differ.
Since `intended_fix` enters the likelihood through `t_stim`, this change in stimulus timing can
affect the fit. The purpose of matching is to compare truncations without also changing the
stimulus-onset distribution.

## Datasets Used By The Algorithm

Inside the fit function three datasets are built:

- `reference_df`: valid trials with `RTwrtStim <= reference_truncate_rt_wrt_stim_s`
- `candidate_df`: valid trials with `RTwrtStim <= truncate_rt_wrt_stim_s`
- `matched_df`: a subset of `candidate_df` sampled so it matches `reference_df`

For the common case:

- `reference_df` corresponds to `115 ms`
- `candidate_df` corresponds to `130 ms`
- `matched_df` is the subsampled `130 ms` dataset used for fitting

## High-Level Matching Strategy

The matching is done in two layers:

1. preserve the composition of important experimental conditions
2. match the `intended_fix` distribution within those condition groups

The condition groups are defined by:

- `ABL`
- `ILD`

So the algorithm does not match globally. It matches within each `(ABL, ILD)` stratum.

This means the final matched sample preserves:

- total number of trials
- number of trials in each `(ABL, ILD)` stratum
- an approximate histogram of `intended_fix` within each `(ABL, ILD)` stratum

## Exact Guarantees

The algorithm guarantees:

- `matched_df` has exactly the same total number of trials as `reference_df`
- `matched_df` has exactly the same number of trials as `reference_df` in every `(ABL, ILD)` stratum
- within each stratum, the matched sample has the same count as the reference sample in each
  `intended_fix` matching bin

So the matched sample is not just similar in size. It is exactly matched in total count and in
stratum counts.

## Step-By-Step Algorithm

### 1. Validate inputs

The code first checks:

- the match column exists in both dataframes
- the grouping columns exist in both dataframes
- there are no `NaN` values in the grouping columns or in the match column

Here:

- match column = `intended_fix`
- grouping columns = `("ABL", "ILD")`

### 2. Split data by `(ABL, ILD)`

The candidate pool is split into strata keyed by `(ABL, ILD)`.

Then the algorithm loops over each `(ABL, ILD)` stratum present in the reference data.

For each stratum:

- `reference_group` = trials from `reference_df`
- `candidate_group` = trials from `candidate_df`

### 3. Check feasibility

For each stratum the algorithm requires:

- the stratum exists in `candidate_df`
- the candidate stratum has at least as many trials as the reference stratum

If not, matching fails with an error.

### 4. Decide how many bins to use

Inside a given stratum, the number of bins is chosen as:

```python
n_bins = min(
    quantile_bins,
    floor(n_reference / min_trials_per_bin),
)
```

with a lower bound of 1.

This means:

- `quantile_bins` is the requested maximum number of bins
- `min_trials_per_bin` stops bins from becoming too sparse

Example:

- if `n_reference = 279`
- `quantile_bins = 10`
- `min_trials_per_bin = 5`

then:

- `floor(279 / 5) = 55`
- so `n_bins = min(10, 55) = 10`

If instead:

- `n_reference = 18`
- `quantile_bins = 10`
- `min_trials_per_bin = 5`

then:

- `floor(18 / 5) = 3`
- so only `3` bins are used

### 5. Build bins from reference quantiles

The bins are not fixed-width bins. They are quantile bins built from the reference sample.

If `n_bins = 10`, the code uses quantiles:

- `0%`
- `10%`
- `20%`
- ...
- `100%`

to define bin edges from the reference `intended_fix` values in that stratum.

This means:

- each bin contains roughly the same number of reference trials
- bin widths in milliseconds are allowed to vary

Dense regions of the distribution get narrow bins. Sparse regions get wider bins.

### 6. Why quantile bins are used

Quantile bins were chosen because they are robust for exact count matching.

If fixed-width bins were used, some bins in sparse parts of the distribution could contain very
few trials. That makes exact matching noisy or impossible.

Quantile bins reduce that problem because each nonempty bin is designed to hold a similar number
of reference trials.

### 7. Why the number of bins is capped

The maximum number of bins is capped to avoid sparse bins.

This is the main reason for the `min_trials_per_bin` rule.

If too many bins are used:

- some bins may contain only 1 or 2 reference trials
- candidate bins may not have enough trials to match them
- the matching becomes brittle and sensitive to random sampling

So the bin cap is not arbitrary. It is there to keep each bin statistically meaningful.

### 8. Count reference trials per bin

After the reference-based bins are built:

- each reference trial is assigned to one bin
- each candidate trial is assigned to those same bin boundaries

Then the algorithm counts how many reference trials fall into each bin.

These reference counts become the matching target.

### 9. Sample the candidate pool to reproduce those counts

For each bin inside a stratum:

- let `target_count` be the number of reference trials in that bin
- find all candidate trials in the same bin
- sample exactly `target_count` candidate trials without replacement

So if the reference stratum has:

- 28 trials in bin 1
- 27 trials in bin 2
- 29 trials in bin 3

then the matched candidate stratum will also have:

- 28 trials in bin 1
- 27 trials in bin 2
- 29 trials in bin 3

### 10. Combine strata back together

After all bins in all strata are processed:

- the sampled strata are concatenated
- the original trial order is restored by sorting indices
- this final dataframe becomes `matched_df`

That `matched_df` is what is used for fitting.

## Fallback Case

If the reference stratum has too few unique `intended_fix` values to form a real set of bins,
the code falls back to:

- uniform random sampling within that stratum
- still preserving the exact stratum count

This is rare, but it prevents failures when quantile edges collapse.

## Interpretation Of The KS Test

The Kolmogorov-Smirnov test is used only as an audit, not as the matching objective.

Two outputs matter:

- KS statistic: smaller is better, closer to `0`
- KS p-value: larger is better; if `p > 0.05`, the distributions are not significantly different
  by the KS test

So:

- a good match means KS statistic decreases
- and KS p-value increases, ideally above `0.05`

The matching algorithm does not optimize KS directly. Instead, it matches bin counts. The KS test
is then used afterward to confirm that the matched distribution is close to the reference.

## What The Diagnostic Plot Shows

The script saves a `2 x 2` diagnostic figure:

- top-left: density before subsampling, `reference` vs `candidate`
- top-right: CDF before subsampling, with KS statistic and p-value
- bottom-left: density after subsampling, `reference` vs `matched`
- bottom-right: CDF after subsampling, with KS statistic and p-value

This figure is intended to visually confirm that:

- before subsampling, the candidate distribution can differ from the reference
- after subsampling, the matched distribution tracks the reference much more closely

## Why This Method Was Chosen

This method was chosen because it is:

- simple
- reproducible
- easy to audit
- compatible with exact count preservation

It is also appropriate for this fit because the likelihood depends on:

- `ABL`
- `ILD`
- `intended_fix`

Matching within `(ABL, ILD)` avoids accidentally changing the trial composition of the fit while
trying to correct the stimulus-onset distribution.

## Possible Alternative

An alternative would be fixed-width bins in milliseconds instead of quantile bins.

That is possible, but it has a drawback:

- fixed-width bins are more likely to become sparse in the tails

If a fixed-width approach is used, it is usually better to also add a rule that merges sparse
adjacent bins until each nonempty bin contains enough reference trials.

For the current implementation, quantile bins were chosen because they are safer and more stable
for exact count matching.
