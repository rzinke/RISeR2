# Deriving PDFs

The following functions take PDFs as inputs, and return (derive) a new PDF as output.

They fall into one of several general categories:
 - Combining multiple observations into one
 - Inferring the value of an unknown event based on known events
 - Calculating a different quantity from others using variable arithmetic


## Mixing multiple observations

RISeR2 provides two functions for combining or mixing multiple observations of the same quantity into a single PDF. Each is used depending on different understanding of the problem.

### Combining variables

One can *combine* variables when multiple independent observations of the same event are captured. The *intersection* of the input PDFs is taken to produce the output.

```bash
riser-combine-variables <observation1>.txt <observation2>.txt <...> -o <output name>.txt
```

This is equivalent to computing the joint probability of two or more random variables in Bayesian statistics. The area of the joint solution indicates the similarity between the input PDFs.

### Pooling variables

*Pooling* is used when the measurements of a quantity might represent different events, or it is uncertain which observation applies to the event of interest. The *mixture* of the input PDFs is taken to produce the output PDF.

```bash
riser-pool-variables <observation1>.txt <observation2>.txt <...> -o <output name>.txt
```


## Inferring unknown values

### Bracketing an event

Sometimes, the value (e.g., age) of an event cannot be directly measured, but can instead be inferred from bracketing measurements on either side of it. RISeR2 provides a function to estimate the value of such a bracketed event:

```bash
riser-create-bracketed-pdf <smaller-valued pdf>.txt <larger-valued pdf>.txt -o <output name>.txt
```

> [!WARNING]
> The ***smaller***- and ***larger***-valued PDFs must be provided in that respective order. PDF construction may fail silently if the larger value is provided first.


## Arithmetic of random variables

RISeR2 supports variable arithmetic. In contrast to variable mixing, which operates along the same value domain as the input variables, arithmetic operations transform the input variables to a new axis.

For this section, we define random variables `X1` and `X2`.

### Addition

> [!NOTE]
> Adding random variables is different from combining or pooling them.

```bash
riser-add-variables <X1>.txt <X2>.txt -o <output name>.txt
```

### Subtraction

Subtracting variables can be used to find the difference in values over some interval. This can be used, for example, in slip rate calculations to determine the difference in displacement $\Delta u$ over some interval, or the difference in age $\Delta t$.

```bash
riser-subtract-variables <X1>.txt <X2>.txt -o <output name>.txt
```

> [!WARNING]
> The order of the input variables matters. The second variable is subtracted from the first.

This function provides the option to limit the domain of the difference to positive-values only.


### Multiplication

Random variables can be multiplied using:

```bash
riser-multiply-variables <X1>.txt <X2>.txt -o <output name>.txt
```

### Division

Random variables can be divided using:

```bash
riser-divide-variables <X1>.txt <X2>.txt -o <output name>.txt
```

> [!WARNING]
> The order of the input variables matters. The first variable is divided by the second.

Dividing variables produces a ratio distribution, which can indicate a rate in slip rate computations. For example, dividing the difference in displacement $\Delta u$ by the difference in age $\Delta t$ produces a slip rate.

