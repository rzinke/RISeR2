# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
