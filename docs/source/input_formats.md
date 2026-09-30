# Input Formats


## Probability density functions

RISeR2 is built around the *probability density function* (PDF), and is designed to handle PDFs of any arbitrary (non-parametric) shape.

### Representation

To accommodate arbitrary shapes, RISeR2 stores the PDFs in memory and on disk as discrete arrays of floating point numbers, with one array defining the values over the PDF domain, and another array defining the probability density at each value. These are stored on disk as plain text (`.txt`) files, with two comma-separated columns for values and probability densities.

PDFs can -- but don't necessarily -- carry metadata, including:
 - `name` -- A descriptive name of the measurement (e.g., sample ID; lab code)
 - `variable_type` -- The physical quantity that the PDF represents (e.g., age; displacement; slip rate)
 - `unit` -- The value unit. Currently, supported base units include years `y` and meters `m`. The base units can have common prefixes `m` (milli), `c` (centi), `d` (deci), `D` (deca), `C` (hecto), `k` (kilo), and `M` (mega).

### Data sources

There are several ways one may create or format a PDF for RISeR2 ingestion.

#### Parametric functions

The RISeR2 library provides support for a number of parametric functions that can be used to create array-based PDFs. PDFs with the shape of a parametric function can be generated and saved to file using the `riser-make-pdf` function. A list of supported parametric functions can be accessed using `riser-make-pdf --help`.

#### Output from other programs

PDFs or PDF-like measurements are sometimes provided by other software packages. For example:
 - [OxCal](https://c14.arch.ox.ac.uk/oxcal.html) provides calibration and analysis for geochronologic ages. RISeR2 provides direct support for converting OxCal outputs to RISeR2-format PDFs via the `riser-calyr-to-age` function.
 - [LaDiCaoz](https://github.com/OlafZielke-EQ/LaDiCaoz_v2) provides a goodness-of-fit (GoF) metric for restoring displaced geomorphic features. The GoF metric can potentially be converted to a PDF with bespoke treatment.

#### Self-definition

Because PDFs are stored as plain text files, a user could define their own PDF by manually writing the desired domain values and probability densities into the file. If the manually defined values are sparse, `riser-interpolate-pdf` can be used to sample the domain more densely and regularly. Metadata items and values are denoted at the top of the file using `#`. For example, a minimal PDF file could look like:

```text
# name: hand
# variable_type: displacement
# unit: m
1.0,0.0
2.0,0.5
3.0,1.0
4.0,0.5
5.0,0.0
```


## Marker config TOML files


