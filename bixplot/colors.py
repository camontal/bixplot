import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.colors as mcolors
import pandas as pd
import warnings

def spaced_palette(cmap_name="Spectral", n_colors=3):
    """Return n_colors maximally spaced colors from a continuous colormap."""
    cmap = sns.color_palette(cmap_name, as_cmap=True)
    return [cmap(x) for x in np.linspace(0, 1, n_colors)]

def darker(color, factor=0.7):
    """Return a darker shade of the given color.
    factor < 1 makes it darker, > 1 makes it lighter.
    """
    rgb = mcolors.to_rgb(color)
    return tuple([c * factor for c in rgb])

def _map_rug_colors(data=None, rug=None, rug_color=None):
    """
    Map rug values to colors.

    Parameters
    ----------
    data : DataFrame, required
        Input dataset. 
    rug : str or None, default=None
        Column name in data providing the values that determine rug colors.
        - If None → no rug mapping performed.
        - If numeric → values mapped to continuous colormap.
        - If categorical → values mapped to discrete palette or dict.
    rug_color : when rug is categorical: str | list | dict | None
                when rug is numerical: str | None
                default=None
        Color specification for rug values:
        - str : Single color (e.g., "black")).
                All rug ticks use the same color.
        - str (palette name) : Name of seaborn colormap (e.g., "viridis", "Set2").
                               Used to map either numeric (continuous) or categorical values.
        - list : Explicit list of colors, only when rug is categorical
        - dict : Explicit mapping from category → color, only when rug is categorical
        - None : Defaults to "dark:salmon_r" for numeric, "Set2" for categorical.

    Returns
    -------
    colors : str | Series
        - Single color string if `rug` is None.
        - Series of colors if `rug` is given.
    palette : matplotlib Colormap or list or None
        - Colormap used (if numeric).
        - List of colors (if categorical).
    norm : matplotlib.colors.Normalize or None
        - Normalization function for numeric rug values.
        - None if not applicable if `rug` is None or categorical

    Notes
    -----
    - Numeric rug values are mapped using continuous colormaps.
    - Categorical rug values are mapped using discrete palettes or dictionaries.
    - Invalid inputs fall back to default palettes ("dark:salmon_r" for numeric,
      "Set2" for categorical).
    - Any unmapped categories in a dict default to "black".
    """

    # --- Case 1: no rug column ---
    if rug is None:
        if rug_color is None:
            return "black", None, None
        try:
            mpl.colors.to_rgba(rug_color) 
            return rug_color, None, None
        except Exception:
            warnings.warn("Invalid rug_color: expected a valid color string."
                          "Black is used instead.", stacklevel=2)
            return "black", None, None

    # --- Case 2: numeric rug values ---
    values = data[rug]
    if pd.api.types.is_numeric_dtype(values):
        try:
            if rug_color is None:
                palette = sns.color_palette("dark:salmon_r", as_cmap=True)
            elif isinstance(rug_color, str):
                palette = sns.color_palette(rug_color, as_cmap=True)
            else:
                raise ValueError
        except Exception:
            warnings.warn("invalid rug_color. Using default.", stacklevel=2)
            palette = sns.color_palette("dark:salmon_r", as_cmap=True)
        norm = mpl.colors.Normalize(vmin=values.min(), vmax=values.max())
        colors =  values.map(lambda v: mpl.colors.to_hex(palette(norm(v))))
        return colors, palette, norm
    
    # --- Case 3: categorical rug values ---
    cats = pd.Categorical(values).categories
    try:
        if rug_color is None:
            palette = spaced_palette('Set2', n_colors=len(cats))
        elif isinstance(rug_color, (str)):
            palette = spaced_palette(rug_color, n_colors=len(cats))
        elif isinstance(rug_color, list):
            palette = sns.color_palette(rug_color, n_colors=len(cats))
        elif isinstance(rug_color, dict):
            for k, v in rug_color.items():
                mpl.colors.to_rgba(v)
            return values.map(rug_color).fillna("black"), None, None
        else:
            raise ValueError
    except Exception:
            warnings.warn("invalid rug_color. Using default.", stacklevel=2)
            palette = spaced_palette("Set2", n_colors=len(cats))
    mapping = dict(zip(cats, [mpl.colors.to_hex(c) for c in palette]))
    return values.map(mapping), None, None


