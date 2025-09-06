# RISeR2

## Purpose
This libarary is designed around the computation of earthquake fault slip rates. A **slip rate** describes the displacement across a fault, averaged over some time interval, i.e.,

$v = \frac{\Delta u}{\Delta t}$

where $v$ is the slip rate, $u$ is the displacement across the fault, and $t$ is the time interval over which the displacement accumulated. In the context of RISeR2, slip rates are defined in the natural record by geologic, geomorphologic, or sedimentologic features for which the displacement can be measured, and the absolute age determined.

Specifically, RISeR2 is designed to compute ***incremental slip rates***, which are slip rates averaged over closed intervals. Ideally, multiple closed displacement-time intervals are defined for a site on a fault.

## How it works
The fundamental object in this library is the **Probability Density Function (PDF)**.

**DatedMarker** comprises a PDF representing *age*, and a PDF representing *displacement*.

A series of DatedMarkers define the displacement-time history of a fault, or slip history.