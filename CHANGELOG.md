# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]
- Metadata override values can be passed to `add_variables`,
  `subtract_variables`, `multiply_variables`, and `divide_variables`.
- Reinstate `combine_variables` cli script.

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
