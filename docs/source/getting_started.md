# Getting Started


## Setup

### System requirements

RISeR2 is developed for UNIX systems. It was not been tested on PC.

It requires **Python 3.12** or above.


### Quick setup (most users)
RISeR2 is available on [PyPI](https://pypi.org/project/riser/). The simplest way to install it is to open a command prompt and simply run:

```
pip install riser
```

To avoid package conflicts, it is recommended to install in a separate environment. One could, for example, use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) or ([micro](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html))[mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).


### Developer setup
If you plan to help develop the library (you totally should!), clone the repository to your local machine

```
git clone https://github.com/rzinke/RISeR2.git
```

change into the repo directory


```
cd RISeR2
```

and install the package


```
pip install -e ".[dev]"
```

If you have any issues with setup, post them to the [Issues](https://github.com/rzinke/RISeR2/issues) page.


## Quick start example

RISeR2 uses probability density functions (PDFs) as both inputs (*age*, *displacement*) and outputs (*slip rate*) for slip rate calculations. The PDFs can be based on either parametric functions approximations (e.g., Gaussian, triangular, etc.) or direct output from calibration programs (e.g., OxCal for ages; LaDiCaoz for displacements).

The simplest way to get started is to approximate one's age and displacment measurements and uncertainties as parametric functions. Let's say we know the age of a displaced geomorphic feature to approximately 5.0 ka, with a 1-*sigma* uncertainty of 1.0 ky. We can create a Gaussian function as follows

```
riser-make-pdf -d gaussian -s 5.0 1.0 --variable-type age --unit ky -o age.txt
```

The function `riser-make-pdf` facilitates the user in creating a RISeR2-style PDF. Passing the `-h` option raises the help menu. The `-d` flag specifies the distribution type (use `-h` for list of available functions). The function parameters (in this case `mu` and `sigma`) are supplied following the `-s` flag. The output name must be explicitly specified via the `-o` flag, and the output PDF filename should end in `.txt`. It is good practice and highly recommended to specify the `variable-type` and `unit`, but not strictly necessary.

We can make a PDF describing the displacement as well. We will use a simple example in which the geologists report the minimum- and maximum-sedimentologically plausible bounds of 22.5 m and 28.0 m, as well as a preferred offset estimate of 25.0 m. These parameters can be represented using a triangular distribution

```
riser-make-pdf -d triangular -s 22.5 25.0 28.0 --variable-type displacement --unit m -o disp.txt
```

The slip rate can then be calculated from the displacement and age PDFs

```
riser-compute-slip-rate --age age.txt --displacement disp.txt -o rate/example
```

where `rate/` is the folder in which the results will be stored, and `example` is the naming system of the output files.

The PDFs produced by each of the above steps can be viewed using the `riser-view-pdf` function, e.g.,

```
riser-view-pdf rate/example_marker_slip_rate.txt
```

Append the `--verbose` flag for a printout of the essential statistics, including the mean and standard deviation of the distribution.
