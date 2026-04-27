# bixplot

`bixplot` is a variation of the classical boxplot designed to detect and display bimodality and multimodality in univariate data. It combines density visualization (like violin plots), summary statistics (like boxplot), and individual observations (like rug plot).

---

## Background

The method is based on the research paper:

“The bixplot: A variation on the boxplot suited for bimodal data”

The classical boxplot assumes unimodality and may fail to reveal meaningful subgroups in the data.  
The bixplot addresses this by:
- testing for unimodality
- if unimodality is rejected, it applies a univariate clustering method that ensures contiguous clusters, meaning that no cluster has members inside another cluster, and such that each cluster contains at least a given number of unique members.
- visualizing clusters as separate components

---

## Installation

pip install bixplot

---

## Quick example

from bixplot import bixplot
import seaborn as sns

df_penguins = sns.load_dataset("penguins")
print(df_penguins.shape)
display(df_penguins.head(2))

fig, axs = plt.subplots(1,2, figsize=(8, 4))
sns.violinplot(data=df_penguins, x='island', y='bill_length_mm', color=R_orange, linecolor="k", ax=axs[0])
axs[0].set_title("violin plot", fontweight='bold', fontsize=10)

bixplot(data=df_penguins, x='island', y='bill_length_mm', ax=axs[1])
axs[1].set_title("bixplot", fontweight='bold', fontsize=10)
plt.tight_layout()

---

## Examples

More examples are available in the `examples/` folder.

---

## Data

The `data/` folder contains datasets used for demonstrations.

See `data/README.md` for full attribution and sources.

---

## License

MIT http://opensource.org/licenses/MIT


