# Creating and Checking PDFs

Essentially all CLI functions in the RISeR2 library require PDFs as fundamental inputs, and many functions produce them as outputs. These functions show how to create, manipulate, and understand PDFs.


## Creating a PDF from a known, parametric distribution

RISeR2 provides support for creating PDFs from parametric functions -- functions with known shape that can be fully described by a finite number of parameters. This library provides support for several such functions that are commonly encountered in the geoscientific literature.

The following command can be used to make a PDF from a parametric function:

```bash
riser-make-pdf -d <function name> -s <function parameters> -o <output name>.txt
```


## Convert PDF from a calendar date to an age

This tool can be used to convert a PDF expressed as a calendar date (AD/BC or CE/BCE) to an age before some reference date. This is essential because RISeR2 expects the time axis of random variables to monotonically increase into the past. With this function, the PDF shape is preserved, but remapped onto a new series of domain values.

```bash
riser-calyr-to-age <date name>.txt -o <output name>.txt
```

Use cases include transforming a calibrated radiocarbon date (from e.g., OxCal) into a RISeR2 PDF.

> [!NOTE]
> The reference date is set to 1950 CE, per some conventions in radiocarbon dating. The reference date can be modified using the `--reference-date` option.

> [!NOTE]
> Whereas the default input unit is years, the default output unit is kilo-years. These can be modified via the `--input-unit` and `--output-unit` options.


## Interpolate a PDF

One might want to resample a PDF onto a new domain axis if, for example
 - Multiple PDFs need to be sampled onto the same set of domain values
 - The domain value spacing needs to be finer

To interpolate or resample a PDF, use:

```bash
riser-interpolate-pdf <pdf name>.txt --xmin <new min domain value> --xmax <new max domain value> --dx <new domain spacing> -o <output name>.txt
```


## Viewing a PDF

As a diagnostic to confirm the correct parameters have been entered, and as a means by which to build intuition for the data set, one may wish to visualize one or more PDFs.

The primary function to do this is:

```bash
riser-view-pdf <pdf name>.txt
```

One may also wish to know, quantitatively, the statistics of a PDF. Passing the `--verbose` (`-v`) option will print a statistical summary of the PDF, including the function's moments, mean, and median.

Furthermore, confidence range(s) of the PDF can be computed and highlighted by passing the `--show-confidence` flag. Add the `-v` flag to print the confidence range.


## Viewing multiple PDFs

One can visualize a data set consisting of multiple PDFs describing, for example, a series of ages, using the function:

```bash
riser-view-pdf-stack <pdfs config file>.toml
```

This requires organizing the PDFs in a TOML-based configuration file.

```toml
["C1"]
"pdf file" = "C1_age.txt"

["C2"]
"pdf file" = "C2_age.txt"
"color" = "dodgerblue"

["C3"]
"pdf file" = "C3_age.txt"
"prior" = "tmp/C3prior_age.txt"
```

As shown in the last example, the config file also allows a PDF to be associated with a *prior* distribution that shows the distribution prior to some analysis (e.g., trimming within an OxCal sequence). See `examples/case_studies/complex_incremental_slip_rates/age_list.toml` for a complete example.
