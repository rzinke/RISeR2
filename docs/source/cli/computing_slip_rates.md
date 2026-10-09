# Computing Slip Rates

RISeR2 is optimized for computing earthquake fault slip rates. Functions are provided for computing long-term average and incremental slip rates, using both closed-form analytical and Monte Carlo sampling methods.

See [input_formats](../input_formats.md) for the structure of the marker configuration file.

Note that, unlike functions designed to output a single PDF, the family of slip rate functions is designed to store outputs in a folder, and the output prefix should not be specified with a `.txt` extension.


## Long-term average rate computation

To compute a long-term average slip rate (based on a single displacement-age marker), use:

```bash
riser-compute-slip-rate
```

The function accepts input PDFs in two forms. The first is as a `marker_config.toml` file with a single marker, or as direct inputs.

Pass a `marker_config.toml` as:

```bash
riser-compute-slip-rate <marker config>.toml -o <output folder>/<output name>
```

or, alternatively, pass the age and displacement inputs directly as

```bash
riser-compute-slip-rate --age <age pdf>.txt --displacement <displacement pdf>.txt -o <output folder>/<output name>
```


## Incremental slip rate computation (analytical approach)

To compute a set of incremental slip rates (between several sets of markers) using the analytical method, use:

```bash
riser-compute-slip-rates <marker config>.toml -o <output folder>/<output name>
```

> [!NOTE]
> The markers must be entered in order of *youngest to oldest*.

One may pass the `--enforce-ordering` flag for additional Bayesian constraint on the slip rate estimates. This trims the marker ages and displacements on the condition that the markers are provided in strict sequential ordering, similar to the Monte Carlo sampling condition `pass non-negative`.

## Incremental slip rate computation (Monte Carlo approach)

To compute a set of incremental slip rates using the Monte Carlo method, use:

```bash
riser-compute-slip-rates-mc <marker config>.toml -o <output folder>/<output name>
```

Every sample draw must pass the condition `pass non-negative`, which requires every marker to be older and more displaced than the one before it such that all incremental rates are non-negative. Draws that fail are rejected as a whole, which conditions all incremental rates on the full stack of markers.

> [!WARNING]
> If the markers are clearly out of order, both commands stop with an error naming the markers. If sampling fails to find the requested number of valid samples, Monte Carlo stops with a warning (or the error “No valid samples were found” if there are none). Check the marker order, or increase `hard_stop`.
