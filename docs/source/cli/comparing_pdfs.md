# Comparing PDFs

RISeR2 offers several functions for quantitatively comparing PDFs, particularly for similarity between two or more PDFs.


## Comparing two random variables

To compare two PDFs using cross-correlation between a reference and seconday variable, use:

```bash
riser-cross-correlate-variables <reference name>.txt <secondary name>.txt -p
```

To compute the Kolmogorov-Smirnov statistic for two PDFs, use

```bash
riser-compute-ks-statistic <pdf1>.txt <pdf2>.txt -p
```

## Comparing two or more random variables

To compute the overlap index between two or more PDFs, use:

```bash
riser-compute-overlap-index <pdf1>.txt <pdf2>.txt <...> -p
```
