# Computing Slip Rates

RISeR2 is optimized for computing earthquake fault slip rates. Functions are provided for computing long-term average and incremental slip rates, using both closed-form analytical and Monte Carlo sampling methods.

See [input_formats](../input_formats.md) for the structure of the marker configuration file.

Note that, unlike functions designed to output a single PDF, the family of slip rate functions is designed to store outputs in a folder, and the output file name should not be specified with a `.txt` extension.


## Long-term average rate computation

To compute a long-term average slip rate (based on a sigle displacement-age marker), use:

```bash
riser-compute-slip-rate
```

The function accepts input PDFs in two forms. The first is as a `marker_config.toml` file, or as direct inputs.

Pass a `marker_config.toml` file as:

```bash
riser-compute-slip-rate <marker config>.toml -o <output folder>/<output name>
```

or, alternatively, pass the age and displacement inputs directly as

```bash
riser-compute-slip-rate --age <age pdf>.txt --displacement <displacement pdf>.txt -o <output folder>/<output name>
```


## Incremental slip rate computation (analytical appraoch)

To compute a set of incremental slip rates (between several sets of markers) using the analytical method, use:

```bash
riser-compute-slip-rates <marker config>.toml -o <output folder>/<output name>
```

## Incremental slip rate computation (Monte Carlo approach)

To compute a set of incremental sip rates using the Monte Carlo method, use:

```bash
riser-compute-slip-rates-mc <marker config>.toml -o <output folder>/<output name>
```
