# Deriving PDFs

The following functions take PDFs as inputs, and returns a new PDFs as output.

They fall into one of several general categories:
 - Combining multiple observations into one
 - Inferring the value of an unknown event based known events
 - Calculating a different quantity from others using variable arithmetic


## Combining multiple observations

RISeR2 provides two functions for combing multiple observations of the same quality into a single PDF. Each is used depending on different understanding of the problem.

### combine_variables
The combine_variables function is used to when multiple independent observations of the same event are captured. The *intersection* of the input PDFs is taken to produce the output.

### pool_variables
The pool_variables function is used when the measurements of a quantity might represent different events, or it is uncertain which observation applies to the event of interest. The *union* of the input PDFs is take. to produce the output PDF.
