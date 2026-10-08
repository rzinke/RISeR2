# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]
### Added
- Introduced `variable_pairs.interpolation` module for interpolating variable
  pair PDFs (e.g., age, displacement) onto common axes.
- Created test module for `variable_pairs.interpolation`.
- `pyproject.toml` `[project.urls]`
- New function `slip_rates/rate_computation/find_slip_rate_tail_cap` to
  determine maximum slip rate value to consider.
- Tests for slip rate reporting, see `test_slip_rate_reporting.py`.
### Changed
- `compute_slip_rates_analytical` can now take the full stack of ages and
  displacements into consideration when calculating incremental slip rates
  when the `enforce_ordering` flag is passed.
- `--enforce-ordering` flag introduced for `compute_slip_rates_analytical`
  command line entry point.
- `--min-rate` / `min_rate` set to default floor of 0.0.
- **Breaking:** `dv` now defaults to `None`, triggering `divide_variables`
  to use 1000 samples across the natural rate.
- **Breaking:** `subtract_variables` can now enforce `limit_positive`.
  Output axis is now cropped to positive values. Enforced by tests.
- Slip rate functions now take `None` sentinel for `max_rate`,
  and determines maximum slip rate to consider based on the
  `find_slip_rate_tail_cap` function. Tests integrated.
- `epsilon` parameter introduced to determine fraction of the positive
  slip rate probability allowed to lie above v_max, control for
  `find_slip_rate_tail_cap`.
- `constrain_above/below` can now crop the output PDF to the constrained range.
- Added tests for `constrain_above/below`.
- **Breaking:** changed `compute_slip_rates_mc` input parameters:
  `pdf_xmin` to `min_rate`; `pdf_xmax` to `max_rate`; `pdf_dx` to `dv` to
  match other slip rate computation functions.
- **Breaking:** Keep original PDF names where variable stays the same:
  `trim_variables`, `constrain_above`, `constrain_below`.
- **Breaking:** Changed default output unit of `calyr_to_age.py`
  from `ky` to `y`.
- Conditions are now reported in slip rate report files.
- **Breaking:** `max_sample_rate` now controls maximum-possible slip rate pick
  in `compute_slip_rates_mc.py`. `max_rate` is the maximum rate for the PDF
  construction.
- Default `n_samples` is now 100 000 (ten times higher than previously).

## [1.6.1] - 2026-10-02
### Added
- Documentation on Read the Docs (getting started, input formats, CLI guide).
- `docs` optional dependency group.
### Fixed
- Parametric function docstring typos; pooling docstring.

## [1.6.0] - 2026-09-28
### Added
- `divide_variables` `dz` can now automatically choose the spacing the spacing
  of the output array.
- Added to `test_variable_arithmetic` to guarantee new `divide_variables`
  behavior.
### Fixed
- A too-coarse `dz` value in `divide_variables` will fail loudly with a
  specific error message.
### Changed
- The limits beyond the natural range of a quotient in `divide_variables` are
  clipped, so output arrays can be shorter than the natural range, but not
  longer.

## [1.5.0] - 2026-09-28
### Changed
- Restructured `examples` folder with top-level folders `cli_reference` and
  `case_studies`. Added `README.md` for structure clarification.
- Update `gitignore` file to reflect new `examples` structure.
- Cleaned up CLI examples and synched with current script names.
- Clearer docstring example for `combine_variables.py`.
### Fixed
- Fixed output units in slip rates scripts and `view_displacement_history`
  to default to input or assume those specified by user.

## [1.4.0] - 2026-09-28
### Added
- Add more optional but explicit arguments to `make_pdf.py`.
### Changed
- Reorganized argument parser groups for all CLI scripts.
- Clarified values from `determine_min_max_limits` are suggestions rather than
  hard requirements.

## [1.3.0] - 2026-09-28
### Added
- Reinstate `combine_variables.py`, `cross_correlate_variables.py`,
  `compute_ks_statistic`, and `compute_overlap_index` cli scripts
  and made executable.
- Reinstate `merge_variables.py` cli script as `pool_variables.py`.
- Reinstate `compute_gap_probabilities.py` as `create_bracketed_pdf.py`.
- Metadata override values can be passed to `add_variables.py`,
  `subtract_variables.py`, `multiply_variables.py`, `divide_variables.py`, and
  `interpolate_pdf.py`.
- Included `riser-compute-slip-rate` in Quick Start example.
### Changed
- Updated CLI parser examples to `riser-...` syntax.
- `compute_slip_rate.py` now accepts direct specification of age and
  displacement PDFs rather than marker TOML file only.
### Fixed
- Arguments parse correctly in `subtract_variables.py`.
- Output now recognized in `interpolate_pdf.py`

## [1.2.0] - 2026-09-25
### Added
- Explicit `name`, `variable_type`, and `unit` keyword-only parameters on
  every PDF-returning function, replacing the previous mix of `**metadata`
  kwargs and inconsistent per-function coverage.
### Changed
- **Breaking:** `compute_slip_rate`, `compute_slip_rates_analytical`, and
  `compute_slip_rates_mc` renamed their rate-step parameter from `dr` to
  `dv`. Update any `dr=` keyword calls to `dv=`.
- **Breaking:** `compute_slip_rates_analytical` now always constrains the
  age difference (`delta_t`) between adjacent markers to positive values,
  regardless of `limit_positive`. `limit_positive` now governs only the
  displacement difference (`delta_u`).

## [1.1.0] - 2026-09-24
- Prior release.
