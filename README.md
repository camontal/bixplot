# bixplot

`bixplot` is a variation of the classical boxplot designed to detect and display bimodality and multimodality in univariate data. It combines density visualization (like violin plots), summary statistics (like boxplot), and individual observations (like rug plot).

---

## Background

The method is based on the research paper:

“The bixplot: A variation on the boxplot suited for bimodal data”
Authors: Camille Montalcini, Peter Rousseeuw

A bixplot extends the violin plot and boxplot by automatically testing each variable for unimodality (via Hartigan’s dip test) and, when multimodality is detected, fitting a constrained k-medoids clustering to identify and separately display the modes. Each variable is rendered as a filled density body, a box-and-whisker summary, and a rug of individual data values. The rug can optionally be colored by an external numeric or factor variable. 

---

## Installation

pip install bixplot

---

## Examples

Examples are available in the `examples/` folder.

---

## Data

The `data/` folder contains datasets used for demonstrations.

See `data/README.md` for full attribution and sources.

---

## License

MIT http://opensource.org/licenses/MIT


