# Rejection sampling for Incremental Slip Rate calculation v.2 &ndash; RISeR2

## Purpose
Earthquake fault slip rates describe the displacement accumulation rate across a fault, averaged over some time interval. Slip rates are determined from geologic, geomorphic, or sedimentary markers recording measureable fault slip, and which can be dated using absolute geochronologic methods. Where multiple dateable displacement markers share a common slip history and record different epochs in that history, one can determine *incremental* slip rates. These average over shorter intervals relative to the oldest displacement, and can show the constancy or variability of fault slip over time.

**RISeR2** is designed to measure incremental fault slip rates based on the dated displacement history of a fault while fully accounting for uncertainties in the ages, displacements, and derived slip rates. It allows enforcement of Bayesian conditions, including the assumption that the fault did not slip backward at any point in its history. The RISeR2 library hosts a set of analytical functions and sampling tools for computing slip rates; formatting, conditioning, and displaying data; and analyzing the results.


## Setup
RISeR2 is optimally used as a set of command line tools, which form an adaptable pipeline for handling and computing probability functions. These scripts have been tested for Linux/UNIX systems. To use these scripts, clone the GitHub repository to a location of your choice, which will be referred to as `RISER2_HOME`. Once cloned, you must append the filepaths to your `$PATH` AND `$PYTHONPATH` variables.

f you are running BASH (`echo $SHELL = bash`), add the following paths to your `~/.bashrc` file for a Linux system or `~/.bash_profile` for a Mac system:
```
RISER2_HOME=<path to repo>

export PATH="${PATH}:${RISER2_HOME}/src/riser/cli"
export PYTHONPATH="${PYTHONPATH}:${RISER2_HOME}/src"
```

If you are running CSH/TCSH (echo $SHELL = (t)csh), add the following paths to your ~/.cshrc or ~/.tcshrc file:
```
set RISER2_HOME=<path to repo>

setenv PATH $PATH\:"$RISER2_HOME/src/riser/cli"
setenv PYTHONPATH $PYTHONPATH\:"${RISER2_HOME}/src"
```

If you are using Windows, it is recommended you set additional environmental variables.


## Top-level Functions and Examples
RISeR2 provides three command line functions for determining slip rates:

 - `compute_slip_rate.py`
 - `compute_slip_rates.py`
 - `compute_slip_rates_mc.py`

Each tool is designed for different scenarios, and to incorporate different assumptions. See [Slip rate determination](#slip-rate-determination).

Example data sets and launch scripts can be found in the `$RISER2_HOME/test` folder.


## Under the Hood
For a single dated measurement of fault displacement, a fault **slip rate** measured since the present day is defined as:

$\bar{v} = \frac{u}{t}$

where $\bar{v}$ is the slip rate averaged over time, $u$ is the displacement across the fault, and $t$ is the time interval over which the displacement accumulated. If multiple dated displacement markers are available along a fault (and all share a common slip history), a displacement-time history or slip history can be determined. For $n$ dated markers, $m = n - 1$ closed intervals in the fault slip history are defined. Slip rates averaged over closed intervals are ***incremental*** **slip rates**, defined as:

$\bar{v}_i = \frac{\Delta u_i}{\Delta t_i}$

where $i$ is one of the $m$ closed intervals, and $\Delta$ is the change in displacement or time within each interval. RISeR2 is specifically designed to compute incremental slip rates.

Often, both the displacement measurement and age of the feature have non-negligible uncertainties. These uncertainties can be expressed as *probability density functions (PDFs)*, which express the *relative likelihood* of any *value* (e.g., time or age, displacement, slip rate, *etc.*) as a function of values. A PDF can take on any shape that adheres to the following criteria:
 - Probability density is never negative. It can be zero or positive over the range of all possible values.
 - The area under the curve is 1.0. I.e., the event is guaranteed to have occurred at some point in the defined values.

Strictly defined, a PDF is absolutely continuous (defined over infinitely many values). RISeR2 uses *discrete PDFs*, represented by a discrete and finite vector of possible values, each with a corresponding probability density. Use of discrete PDFs is essential because RISeR2 is designed to work with non-parametric functions (such as calibrated radiocarbon dates) that cannot be described by closed-form equations.

**`PDFs`** are the fundamental object in the RISeR2 library. All ages, displacements, and slip rates are expressed as PDFs with the above properties. The PDF can be given a descriptive *name*, and a description of the *variable-type*, e.g., "age", "displacement*, *slip rate*, *etc.* The probability density of any value within the vector of PDF values can be determined by linear interpolation between defined probability densities, and values outside the defined vector are assigned zero probability density. 

Data points defining displacement-time history of a fault are encoded as a **`DatedMarker`**, comprising a PDF representing *age*, and a PDF representing *displacement*. Like a PDF, a DatedMarker can carry a descriptive name.

### Slip rate determination

`compute_slip_rate.py` is used to determine the slip rate of a fault since present day, based on a single DatedMarker. It uses the analytical formulation (a weighted convolution) of Bird (2007).

`compute_slip_rates.py` is used to compute incremental slip rates between multiple DateMarkers using convolution-based analytical formulas. It assumes that neither the change in age between markers $\Delta t$, nor the change in displacement $\Delta u$ are negative.

`compute_slip_rates_mc.py` invokes Monte Carlo-style sampling of the ages and displacements of multiple DateMarkers in a slip history. The age and displacement PDFs are sampled according to their non-parametric forms using the **Probability Inverse Transform (PIT)** method. Multiple conditions can be enforced on whether the random samples are valid and accepted, or invalid and rejected in the process of **rejection sampling**. Like Gold & Cowgill (2012), `compute_slip_rates_mc.py` enforces the assumption that the fault did not slip backwards (inversely to its overall kinematics) at any point in its history. This can be especially important in cases where ages and/or displacements uncertainties overlap. A maximum-allowable slip rate may also be provided, to avoid possibilities in which the fault slipped unrealistically fast over multiple earthquake cycles.


## References
 - Bird, P. (2007), Uncertainties in long-term geologic offset rates of faults: General principles illustrated with data from California and other western states, *Geosphere, 3*, 577–595, https://doi:10.1130/GES00127.1.
 - Gold, R. D., & Cowgill, E. (2011). Deriving fault-slip histories to test for secular variation in slip, with examples from the Kunlun and Awatere faults. *Earth and Planetary Science Letters, 301*, 52–64. https://doi.org/10.1016/j.epsl.2010.10.011
 - Zinke, R., Dolan, J. F., Rhodes, E. J., Van Dissen, R., & McGuire, C. P. (2017). Highly variable latest Pleistocene–Holocene incremental slip rates on the Awatere fault at Saxton River, South Island, New Zealand, revealed by lidar mapping and luminescence dating. *Geophysical Research Letters, 44*. https://doi.org/10.1002/2017GL075048
 - Zinke, R., Dolan, J. F., Rhodes, E. J., Van Dissen, R., McGuire, C. P., Hatem, A. E., et al. (2019). Multimillennial incremental slip rate variability of the Clarnce fault at the Tophouse Road site, Marlborough Fault System, New Zealand. *Geophysical Research Letters, 46*, 717–725. https://doi.org/10.1029/2018GL080688
 - Zinke, R., Dolan, J. F., Rhodes, E. J., Van Dissen, R. J., Hatem, A. E., McGuire, C. P., et al. (2021). Latest Pleistocene–Holocene incremental slip rates of the Wairau fault: Implications for long-distance and long-term coordination of faulting between North and South Island, New Zealand. *Geochemistry, Geophysics, Geosystems, 22*, e2021GC009656. https://doi.org/10.1029/2021GC009656


## Citation
 - Zinke, R., Dolan, J. F., Rhodes, E. J., Van Dissen, R. J., Hatem, A. E., McGuire, C. P., et al. (2021). Latest Pleistocene–Holocene incremental slip rates of the Wairau fault: Implications for long-distance and long-term coordination of faulting between North and South Island, New Zealand. *Geochemistry, Geophysics, Geosystems, 22*, e2021GC009656. https://doi.org/10.1029/2021GC009656
 - Zenodo: https://zenodo.org/records/4733235. https://doi.org/10.5281/zenodo.4733235


## Disclaimer
This is research code. The author(s) assumes no responsibility for any errors in the methods or scripts, or any damages resulting from the code's use.