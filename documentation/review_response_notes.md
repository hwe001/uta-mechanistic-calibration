# Notes from the internal senior review (2026-10-01)

Checked against the saved outputs and data; see `code/extended_analysis/reviewer_response.py` and `results/reviewer_response.json`.

- **Residual SD corrected.** The benchmark's within-woman residual SD is 0.197 (variance 0.0390), not 0.28 (0.28 is roughly the SD of the marginal residuals, i.e. random-intercept plus residual variance). ICC after gestational age 0.53; null-model ICC 0.26. Variance of log-PI: gestational age 32%, between women 36%, within women 32%. The suggestion that about 70% of the variance is between women is not supported.
- **Bootstrap unit.** The trajectory confidence intervals previously came from resampling the 13 bin medians; they are replaced by a woman-level bootstrap (300 resamples, bins rebuilt). Widths are similar. Figure 9a bands still come from the bin-level bootstrap and should be regenerated if the figure is to match the text exactly.
- **Sobol indices depend on how failed runs (6%) are filled.** Failures concentrate at the smallest radial and uterine radii. Radial and uterine radius stay the two largest indices under all fills; terminal resistance never exceeds 0.16. Saved samples (`results/extended_analysis/sobol_*.npy`) allow recomputation.
- **Jacobian.** Condition number varies with the evaluation point (25 to 2,557; 369 at baseline). Noise-weighted, the second singular value is at most 0.40 across the points tried.
- **Synthetic recovery.** The primary run used three starts; a 20-start repeat (`results/identifiability_20starts`) gives the same picture.
- **Bin width.** One-, two- and four-week bins change the fitted radial radius by less than 1%.

## Topology check and Figure 8 (2026-10-01)
- `code/extended_analysis/topology_scan.py` (`results/topology_scan.json`): with the A-V shunt radius at 0.05 mm, scanning terminal resistance 0.2-100 raises PI from 0.48 to 10.7 at the baseline radial radius; with the baseline (0.2 mm) or 3.0 mm shunt PI stays within 0.39-0.54. The earlier statement that distal resistance cannot reach early-pregnancy PI therefore holds only with the shunt at its baseline setting.
- `code/extended_analysis/make_fig8_two_panel.py` adds panel (b): S/D of each axis relative to the radial axis at matched PI. The axes agree up to PI 1.2 and separate above it (uterine +6.7% at 1.3, +9.6% at 1.4; arcuate about +2%).
