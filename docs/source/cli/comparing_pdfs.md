# Comparing PDFs

RISeR2 offers several functions for quantitatively comparing PDFs, particularly for similarity between two or more PDFs.

> [!NOTE]
> Throughout, passing the `--verbose` (`-v`) flag is strongly recommended and may be necessary to view the comparison metric results. Passing the `--plot` (`-p`) flag is also recommended to functions for which the plotting option is available.


## Comparing two random variables

To compare two PDFs using cross-correlation between a reference and secondary variable, use:

```bash
riser-cross-correlate-variables <reference name>.txt <secondary name>.txt -v -p
```

To compute the Kolmogorov-Smirnov statistic for two PDFs, use

```bash
riser-compute-ks-statistic <pdf1>.txt <pdf2>.txt -v -p
```

## Comparing two or more random variables

To compute the overlap index between two or more PDFs, use:

```bash
riser-compute-overlap-index <pdf1>.txt <pdf2>.txt <...> -v -p
```
