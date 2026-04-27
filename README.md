# bixplot

`bixplot` is a variation of the classical boxplot designed to detect and display bimodality and multimodality in univariate data. It combines density visualization (like violin plots), summary statistics (like boxplot), and individual observations (like rug plot).

---

## Background

The method is based on the research paper:

“The bixplot: A variation on the boxplot suited for bimodal data”
Authors: Camille Montalcini, Peter Rousseeuw

The classical boxplot assumes unimodality and may fail to reveal meaningful subgroups in the data.  
The bixplot addresses this by:
- testing for unimodality
- if unimodality is rejected, it applies a univariate clustering method that ensures contiguous clusters, meaning that no cluster has members inside another cluster, and such that each cluster contains at least a given number of unique members.
- visualizing clusters as separate components

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


