Extended analyses added in the CMPB revision (rev12). Run from this folder after the main pipeline.
Order: other_axes.py -> manifold.py (axis_grids.csv, matched_PI.csv) -> extended_mech.py (uterine/arcuate calibration and cross-validation)
       -> control_cv.py (solver-free controls) -> sobol.py (global sensitivity) -> sd_check.py / sd_individual.py (out-of-sample S/D)
       -> boot_ci.py -> make_figs.py (figures 8-10).
Paths at the top of each script point to the UtA project folder; edit `base` if you move it.
Outputs are in results/extended_analysis/.
