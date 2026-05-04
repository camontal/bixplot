#basics
import os
import warnings
import math
import numpy as np
import pandas as pd
import sys

#stats
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity
from diptest import diptest
import pulp # linear programming package

from sklearn_extra.cluster import KMedoids 
from matplotlib.collections import LineCollection    #vectorized the plot for rug inside outside            

#plotting
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .pamc1d import pamc1d
from .colors import spaced_palette, _map_rug_colors, darker

def bixplot(data=None, x=None, y=None, orient='v', group_order=None, width=0.8, 
            diplevel=0.01, minN=15, clusMinN=3, kmax=5, maxit=500, stand=False, verbose=False, 
            hue=None, hue_color=None, hue_order=None, split=True, hue_legend=True, legend_position=None,
            mode_color=None, mode_legend=False, mode_color_by_suffix=True,
            rug_hue=None, rug_color=None, rug_legend=False, jitter=False, rug_linewidths=1, showrug=True,
            rug_colorbarthickness=30, rug_colorbarheight=1, rug_colorbarposition=None,rug_length=0.12,
            rug_outer_color=None, rug_outer_linewidths=1.5,
            box_color='black', showbox=True, box_linewidth=1, box_width=1,
            density_color='mode', showdensity=True, density_alpha=0.5, density_borderlinewidth=1, 
            density_norm='width', kernel= 'gaussian', bandwidth='scott', cut=3, cutmin=None, cutmax=None, bigN=500, 
            undo_constrainkmax=False, random_state=None, ax=None):
    """    
    Bixplot: distribution plot combining density curves, boxplots, and rug plots, with support 
    for multimodal clustering. Densities and boxplots are shown only for groups with at least four unique values; 
    groups with fewer are represented by individual points.
 
    Parameters, 
    We refer to ‘body’ as the filled area that visualizes the estimated 
    probability density of the data (or of each mode when multimodality 
    is detected).
    
    ----------

    data : DataFrame, Series, dict, array, or list of arrays
        Input dataset for plotting.
        - DataFrame (long-form): provide column names in `x` and `y`.
        - DataFrame (wide-form): each column is a distribution, converted internally to long-form.
        - Series: passed directly as `x` or `y`.
        - dict: keys are variable names, values are arrays/lists of observations.
        - array or list of arrays: treated as distributions (wide-form).
        
        If both `x` and `y` are None, `data` is interpreted as wide-form.
        Otherwise, `data` is expected to be long-form and `x`, `y` specify the relevant columns.
    x, y : str or None
        Column names for x and y axes. If both omitted, data is interpreted as wide-form (i.e., a dataframe where each 
        column is one variable (distribution you want to plot) and each row is one observation). Otherwise it is 
        expected to be long-form.
    orient : {'v', 'h'}, default='v'
        Plot orientation: 'v' for vertical, 'h' for horizontal.
    group_order : list, optional
        Ordering of grouping axis
    width : float, default=0.8
        Width of the body
        
    diplevel : float, default = 0.01
        the level of Hartigan's dip test for unimodality of a variable. Only 
        when this test rejects unimodality, that is, its p-value is at most diplevel,
        do we search for clusters. Defaults to 0.01. Setting it higher can lead to
        more clusters, and setting it lower to fewer clusters.
    minN : int, default = 15
        the minimum number of observations required
        per potential cluster. The maximum number of 
        clusters searched is limited to n/minN, where n is
        the number of non-missing values. If n < 2 ∗ minN, 
        clustering is not attempted. 
    clusMinN : int, default = 3
        The clustering is constrained so that any cluster must contain at 
        least clusMinN unique values. When a variable has fewer values 
        than this, only points are drawn.
    kmax : int, default=None    
        Maximum number of clusters to test. It will be truncated to 7 
        internally. If NULL, it is set to min(floor(n/minN), 5). 
        When setting kmax=1 all variables are considered as single 
        clusters, making the display resemble a violin plot.
    maxit : int, default=500
        Maximum number of iterations in the constrained clustering loop.
    verbose : bool, defualt = False
        If True, will print intermediate steps
        
    hue : str, optional
        Categorical variable for additional grouping. Supports >2 categories (forces split=False).
    hue_color : str | list | dict, optional
        Defining the color of each hue levels. Accepts seaborn palette name, list of colors (will be recycled as needed), or dict mapping.
    hue_order : list, optional
        Explicit ordering of hue categories. Missing categories will be appended alphabetically.
    split : bool, default=True
        If True and hue has exactly 2 levels, densities are split symmetrically around center.
    hue_legend : bool, default=True
        If True, adds a legend for hue levels mapping the colour to the hue levels.
    legend_position : str | (float, float), optional
        Location of combined legend (modes, hue, rug). Accepts matplotlib legend locations
        (e.g. 'upper right') or bbox_to_anchor coordinates.

    mode_color : str | list | dict, optional
        Colors for modality clusters. Accepts seaborn palette, list of colours (will be recycled as needed), or dict mapping.
    mode_legend : bool, default=False
        If True, adds legend for detected modalities (clusters).
    mode_color_by_suffix : bool, default=False
        If True, assign colors based only on the modality suffix (e.g., "_0", "_1"),
        so that the same clustering suffix across different groups shares the same color.
        If False, each full modalityID (group + suffix) gets its own color.        
        
    rug_hue : str or None, default=None
        Column name in data providing the values that determine rug colors.
        - If None → no rug mapping performed.
        - If numeric → values mapped to continuous colormap.
        - If categorical → values mapped to discrete palette or dict.
    rug_color : when rug is the column name of categorical data: str | list | dict | None
                when rug is the column name of numerical data: str | None
                when rug is None: str
                default=None
        Color specification for rug values:
        - str : Single color (e.g., "black")).
                All rug ticks use the same color.
        - str (palette name) : Name of seaborn colormap (e.g., "viridis", "Set2").
                               Used to map either numeric (continuous) or categorical values.
        - list : Explicit list of colors, only when rug is categorical. will be recycled as needed
        - dict : Explicit mapping from category → color, only when rug is categorical. 
                 Groups that are not given as a key in this dicitonary aure by default set to black
        - None : Defaults to "dark:salmon_r" for numeric, "Set2" for categorical.
    rug_legend : bool, default=False
        If True, adds legend for rug values, mapping the colour ro the rug levels. If numeric, displays as colorbar.
    jitter : bool, default=False
        Add random jitter to rug positions to reduce overlap.
    rug_linewidths : float, default=1
        Line width of rug ticks.
    showrug : bool, default=True
        Whether to show rug plot. Force to True when rug column is provided
    rug_colorbarthickness : int, default=30
        Aspect ratio of rug colorbar (larger → thinner bar).
    rug_colorbarheight : float, default=1
        Relative height of rug colorbar (1 = full axis).
    rug_colorbarposition : [x, y, w, h], optional
        Manual position of rug colorbar in figure coordinates.
    rug_length : int
        Size of the rug
    rug_outer_color, a color name or None, default=None
        Specifies the color(s) of the part of the ruglines outside 
        of the body. If None, the same color as inside the body is used
    rug_outer_linewidths : float, default=1.5,
        specifies the width of the the ruglines outside 
        of the body
        
    box_color : str | {'mode','hue'}, default='black'
        Color mapping for boxplots. Can follow mode colors, hue colors, or fixed color
    showbox : bool, default=True
        Whether to display boxplots.
    box_linewidth : float, default=1
        Line width for all boxplot elements.
    box_width : float, default = 1
        Width of boxplots
        
    density_color : str | {'mode','hue'}, default='mode'
        Color mapping for densities. Can follow mode colors, hue colors, or fixed string.
    showdensity : bool, default=True
        Whether to display density estimates.
    density_alpha : float, default=0.5
        Transparency of density fill. Set to 0 to show only border.
    density_borderlinewidth : float, default=1
        Line width of density border.
    density_norm: 'area'| 'count' | 'width'
        Method that normalizes each density to determine the violin’s width. 
        If area, each violin will have the same area. 
        If count, the width will be proportional to the number of observations. 
        If width, each violin will have the same width.
    kernel: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 
            'linear', 'cosine', default='gaussian'
        The kernel for use by klearn.neighbors.KernelDensity()
    bandwidth: float or 'scott', 'silverman', default='scott'
        Defines the bandwidth of the kernel.
        If bandwidth is a string, one of the estimation methods is implemented.
    cut : int, default=3
        controls how far the kernel density estimate extends beyond the 
        minimum and maximum observed data. If 0 the density is clipped 
        to the range of the data. Denoting a variable by y, the density is only
        constructed from min(y) - cut*bw to 
        max(y) + cut*bw where bw is the numeric 
        bandwidth.        
    cutmin : int, default=None
        if specified, the densities of all variables
        and modes cannot go below the value cutmin.
    cutmax : int, default=None
        if specified, the densities of all variables
        and modes cannot go above the value cutmax.

    random_state : int, optional
        Random seed for reproducibility.
    bigN : int, default=500
        When a variable has over bigN non-NA values,
        we sample bigN values from it without 
        replacement, to save computation time.
    undo_constrainkmax: bool, default = False
        If True, prevents truncation of kmax to 7 internally.
    ax : matplotlib Axes, optional
        Axis on which to draw. 

    Returns
    -------
    dico_xhue_results : dict
        Dictionary with clustering and dip test results per group.
    DataFrame
        Original data with an additional `modalityID` column for cluster assignment.
    """
 
    if ax is None:
        ax = plt.gca()
        
    if data is None:
        raise ValueError("`data` must be provided")
        
    if not showdensity and not showbox:
        warnings.warn("Both `showdensity` and `showbox` are False. "
                      "Defaulting to boxplot display.", stacklevel=2)        
        showbox = True        

    if hue is not None:
        if box_color!='hue' and density_color!='hue':
            warnings.warn( "Hue is provided, but neither density nor boxplot is "
                           "mapped to hue. Defaulting to hue-colored boxplots.",
                           stacklevel=2)
            box_color = 'hue'
        if box_color == 'hue':
            showbox = True
        if density_color == 'hue':
            showdensity = True
            
    if rug_hue is not None and not showrug:
        warnings.warn( "`rug` is specified but `showrug=False`. Reverting "
                       "to `showrug=True`.", stacklevel=2)
        showrug = True
        
    if mode_legend and density_color!='mode' and box_color!='mode':
        mode_legend=False

    # ------------------------------------------------------------------ #
    # Convert data to dataframe
    # ------------------------------------------------------------------ #
    if not isinstance(data, pd.DataFrame):
        
        # Case 0: pandas Series → wrap as DataFrame
        if isinstance(data, pd.Series):
            arr = data.to_numpy()
            colname = data.name if data.name is not None else ""
            data = pd.DataFrame(arr, columns=[colname])
            if x is not None or y is not None:
                warnings.warn("When passing a Series, both `x` and `y` must be None.", 
                              stacklevel=2)
            x, y = None, None    
        
        # Case 1: wide-form dict of arrays
        elif isinstance(data, dict):
            data = pd.DataFrame(data)
            
        # Case 2: list, tuple, numpy array
        elif isinstance(data, (list, tuple, np.ndarray)):
            arr = np.asarray(data)
        
            # flatten (n,1) → (n,) (i.e., arr.ndim = 1), but keep (n,m) for m>1
            if arr.ndim == 2 and arr.shape[1] == 1:
                arr = arr.ravel()
                
            # single column DataFrame
            if arr.ndim == 1:
                data = pd.DataFrame(arr, columns=[""])
            #if (n,m)
            elif arr.ndim == 2:
                data = pd.DataFrame(arr, columns=["col%d"%i for i in range(arr.shape[1])])
            else:
                raise ValueError("Array input must be 1D or 2D.")
        
            if x is not None or y is not None:
                warnings.warn("When passing `data` not as a dataframe, both `x` and `y`"
                              " must be None.", stacklevel=2)
            x, y = None, None
            
        else:
            raise ValueError("`data` must be a pandas DataFrame, pandas series, dict,"
                             " array, list, or tuple.")

    # ------------------------------------------------------------------ #
    # Infer orientation, axis, and split
    # ------------------------------------------------------------------ #
    if orient not in {None, "v", "h"}:
        raise ValueError("`orient` must be either None, 'v', or 'h'.")

    x_is_num = pd.api.types.is_numeric_dtype(data[x]) if x is not None else False
    y_is_num = pd.api.types.is_numeric_dtype(data[y]) if y is not None else False
    
    # Case 1: neither x nor y is given → wide-form
    if y is None and x is None:
        numeric_cols = data.select_dtypes(include=np.number).columns
        if numeric_cols.empty:
            raise ValueError("data´ must contain at least one numeric column.")     
        data = pd.melt(data, value_vars=numeric_cols, var_name="", value_name="values")
        #if user have not specific the orientation, then use vertical
        orient = orient or "v"
        x, y = ("", "values") if orient == "v" else ("values", "")
            
    # Case 2: only x is numeric
    elif x_is_num and not y_is_num:
        if orient=='v':
            warnings.warn("Vertical orientation ignored with only `x` specificied as" 
                        "numeric while `y` is not.", stacklevel=2)
        orient = "h"
        if y is None:
            data = pd.melt(data, value_vars=[x], var_name="", value_name="values")
            x, y = 'values', ''
        
    # Case 3: only y is numeric
    elif y_is_num and not x_is_num:
        if orient=='h':
            warnings.warn("Horizontal orientation ignored with only `y` specificied"
                          "as numeric while `x` is not.", stacklevel=2)
        orient = "v"
        if x is None:
            data = pd.melt(data, value_vars=[y], var_name="", value_name="values")
            x, y = '', 'values'          
             
    # Case 4: both x and y numeric
    elif x_is_num and y_is_num:
        #if user have not specific the orientation, then use vertical
        orient = orient or "v"     
            
    else:
        raise ValueError("At least one of x or y must be numeric")

    #Standardize notation across orientations
    gr, val, flip = (x, y, False) if orient == "v" else (y, x, True)
    group_vars = [gr]
    kde_var = val
    if hue is not None:          
        group_vars.append(hue)    
        
    # Check split
    if split and hue is None:
        split = False
    if split and (data.groupby(group_vars[0])[hue].nunique() > 2).any():
        warnings.warn("Split ignored when more than two unique values of "
                      "hue per groups.", stacklevel=2)
        split = False           
    if split and (data.groupby(group_vars[0])[hue].nunique() == 1).all():
        warnings.warn("Split ignored when all groups have only one unique"
                      "hue value.", stacklevel=2)
        split = False   
        
    # ------------------------------------------------------------------ #
    # Preprocessing & Fit modalities
    # ------------------------------------------------------------------ #
    data['unique_idbixplot'] = range(len(data)) #to merge back at the end
    init_data = data.copy()
    data = data.dropna(subset=[kde_var]+group_vars).copy()  
    if any(data.groupby(group_vars, observed=True).size() > bigN):
        warnings.warn(
            "Some groups were subsampled to %d values for efficiency see bigN parameter). "
            "The minimum and maximum values in each group were always retained."
            % bigN, stacklevel=2)
        li_subsets = []
        for _, d in data.groupby(group_vars, observed=True):
            if len(d) <= bigN:
                li_subsets.append(d)
                continue
            min_idx = d[kde_var].idxmin()
            max_idx = d[kde_var].idxmax()
            li_extreme_idx = [min_idx, max_idx]
            d_remaining = d.drop(index=li_extreme_idx)
            n_remaining = bigN - len(li_extreme_idx)
            d_sample = d_remaining.sample(n=n_remaining, random_state=random_state)
            d_sub = pd.concat([d.loc[li_extreme_idx], d_sample], axis=0)
            li_subsets.append(d_sub)
        data = pd.concat(li_subsets, axis=0).reset_index(drop=True)

    dico_xhue_results, data = bixplot_methods(data=data, group_vars=group_vars, 
                                              kde_var=kde_var,
                                              kmax=kmax, 
                                              minN=minN, 
                                              clusMinN=clusMinN,
                                              diplevel=diplevel, stand=stand, 
                                              maxit=maxit, 
                                              verbose=verbose, 
                                              undo_constrainkmax = undo_constrainkmax,
                                              random_state=random_state)    

    # ------------------------------------------------------------------ #
    # Define order of hue and grouping variables
    # ------------------------------------------------------------------ # 
    # If order given by user, then keep only those that exist in the data
    # and add missing categories at the end in alphabetical order
    if hue is not None:
        hu = data[hue].unique()
        if hue_order is None:
            hue_order = sorted(hu)
        else:
            hue_order = [h for h in hue_order if h in hu]
            remaining = sorted(set(hu) - set(hue_order))
            hue_order.extend(remaining)     
    gu = data[gr].unique()
    if group_order is None:
        group_order = list(gu)
    else:
        group_order = [c for c in group_order if c in gu]
        remaining = sorted(set(gu) - set(group_order))
        group_order.extend(remaining) 
        
    # ------------------------------------------------------------------ #
    # Positions of groups and modes
    # ------------------------------------------------------------------ #    
    dico_groupID_positions = {g: i for i, g in enumerate(group_order)}
    dico_modeID_positions = {}

    # If several axis within each values on the axis, position hue
    # levels within each group and assign associated mode.
    # First, determine the center of each hue (within each group)
    n_hue = 1 if hue is None else data.groupby(group_vars[0])[hue].nunique().max()
    step = width / n_hue
    offsets = np.linspace(-width / 2 + step / 2, width / 2 - step / 2, n_hue)
    if hue is not None and not split:
        for groupID, df_ in data.groupby(gr):
            hue_levels = [h for h in hue_order if h in df_[hue].unique()]
            for offset, hueID in zip(offsets, hue_levels):
                pos = dico_groupID_positions[groupID] + offset  
                for modeID in df_[df_[hue]==hueID]['modalityID'].unique():
                    dico_modeID_positions[modeID] = pos
                    
    # Else position modes directly at the group center
    else:
        for modeID, groupID in data.groupby(["modalityID", gr], observed=False).groups:
            dico_modeID_positions[modeID] = dico_groupID_positions[groupID]
    
    # ------------------------------------------------------------------ #
    # ´Sign´ assignment to control density curve placement around positions
    # ------------------------------------------------------------------ #   
    dico_modeID_signs = {modeID:[-1,1,0] for modeID in data['modalityID'].unique()}
    if split:
        for groupID, df_group in data.groupby(gr):
            hueID_group = [h for h in hue_order if h in df_group[hue].unique()]
            signs = {hueID_group[0]: [0,-1,-1]} #first element in the split
            if len(hueID_group)==2:
                signs[hueID_group[1]] = [0,1,1] #second element in the split if any
            # Assign to each modeid in this group the correct signs
            for modeID, hueID in df_group.groupby(["modalityID", hue]).groups:
                dico_modeID_signs[modeID] = signs[hueID] 

    # ------------------------------------------------------------------ #
    # Define colors
    # ------------------------------------------------------------------ #
    # --- Rug color ---
    data['rug_colorid'], rug_palette, rug_norm = _map_rug_colors(data=data, rug=rug_hue, 
                                                                 rug_color=rug_color)
    # --- Hue color ---
    hue_pal = 'plasma'
    if hue is not None:
        hue_color_map = None
        try:
            if hue_color is None:
                hue_palette = spaced_palette(hue_pal, n_colors=len(hue_order))
            elif isinstance(hue_color, (str)):
                hue_palette = spaced_palette(hue_color, n_colors=len(hue_order))
            elif isinstance(hue_color, list):
                hue_palette = sns.color_palette(hue_color, n_colors=len(hue_order))
            elif isinstance(hue_color, dict):
                # validate all colors
                [mpl.colors.to_rgba(v) for v in hue_color.values()]
                if not set(hue_order).issubset(hue_color.keys()): 
                    missing = set(hue_order) - set(hue_color.keys())
                    warnings.warn(f"`hue_color` missing some hue levels. Missing: {missing}.",
                                  stacklevel=2)
                    hue_palette = spaced_palette(hue_pal, n_colors=len(hue_order))
                else:
                    hue_color_map = hue_color
            else:
                raise ValueError
        except Exception:
            warnings.warn("Invalid hue_color, falling back to default.", stacklevel=2)
            hue_palette = spaced_palette(hue_pal, n_colors=len(hue_order))       
            
        if hue_color_map is None:
            hue_color_map = dict(zip(hue_order, [mpl.colors.to_hex(c) for c in hue_palette]))        

    # --- Mode color ---
    modality_levels = sorted(data['modalityID'].unique())
    mode_color_map = None
    mode_unique = data['modalityID'].unique()
    if mode_color_by_suffix:
        # Extract only the suffixes (_0, _1, …) for color assignment
        modality_levels = sorted(set([m.split("_")[-1] for m in modality_levels]))
    try:
        if mode_color is None:
            mode_palette = spaced_palette("Spectral", n_colors=len(modality_levels))
        elif isinstance(mode_color, (str)):
            mode_palette = spaced_palette(mode_color, n_colors=len(modality_levels))
        elif isinstance(mode_color, list):
            mode_palette = sns.color_palette(mode_color, n_colors=len(modality_levels))
        elif isinstance(mode_color, dict):
            # validate all colors
            [mpl.colors.to_rgba(v) for v in mode_color.values()]
            if not set(modality_levels).issubset(mode_color.keys()):
                missing = set(modality_levels) - set(mode_color.keys())
                warnings.warn(f"`mode_color` missing some modality levels. Missing: {missing}."
                              "Falling back to Spectral.",stacklevel=2)
                mode_palette = spaced_palette("Spectral", n_colors=len(modality_levels))              
            else:
                if mode_color_by_suffix:
                    mode_color_map = {m: mode_color[m.split("_")[-1]] for m in mode_unique}
                else:
                    mode_color_map = mode_color
        else:
            raise ValueError
    except Exception as e:
        warnings.warn("Invalid mode_color. Falling back to Spectral.", stacklevel=2)
        mode_palette = spaced_palette("Spectral", n_colors=len(modality_levels))

    if mode_color_map is None:
        if mode_color_by_suffix:
            suffix_to_color = dict(zip(modality_levels, [mpl.colors.to_hex(c) for c in mode_palette]))
            mode_color_map = {m: suffix_to_color[m.split("_")[-1]] for m in mode_unique}
        else:
            mode_color_map = dict(zip(modality_levels, [mpl.colors.to_hex(c) for c in mode_palette]))
    
    # ------------------------------------------------------------------ #
    # Define density and box color strategy
    # ------------------------------------------------------------------ #
    def _color(type_color, hue, name, default):
        if type_color == "hue" and hue is None:
            warnings.warn("`%s='hue'` but hue is None. Using %s."
                          % (name, repr(default)), stacklevel=2)
            return default
        if type_color is None:
            return default
        elif type_color == "hue":
            return "hue"
        elif type_color == "mode":
            return "mode"
        elif isinstance(type_color, str) and mpl.colors.is_color_like(type_color):
            return type_color
        else:
            warnings.warn("Invalid `%s`. Falling back to %s."
                          % (name, repr(default)), stacklevel=2)
            return default
    density_color = _color(density_color, hue, "density_color", "mode")
    box_color = _color(box_color, hue, "box_color", "black")       

    # ------------------------------------------------------------------ #
    # Plot the bixplot
    # ------------------------------------------------------------------ #
    handles, labels = [], []   # collect for legend
    
    if split:
        rug_length /= 2 
        
    def compute_bandwidth(vals, bandwidth="scott"):      
        vals = np.asarray(vals).ravel()
        n = len(vals)
        std = np.std(vals, ddof=1)
        if bandwidth == "scott":
            return std * n ** (-1 / 5)
        elif bandwidth == "silverman":
            iqr = np.subtract(*np.percentile(vals, [75, 25]))
            scale = min(std, iqr / 1.34)
            return 0.9 * scale * n ** (-1 / 5)
        elif isinstance(bandwidth, (float, int)):
            return float(bandwidth)
        else:
            raise ValueError("Unsupported bandwidth: %s" % bandwidth)

    def _scaling(v, scaled, bandwidth):
        vals = np.asarray(v).ravel()
        if len(vals) < clusMinN:
            return 0  
        bw = compute_bandwidth(vals, bandwidth)
        vmin = vals.min() - cut * bw if cutmin is None else max(cutmin, vals.min() - cut * bw)
        vmax = vals.max() + cut * bw if cutmax is None else min(cutmax, vals.max() + cut * bw)
        val_grid = np.linspace(vmin, vmax, 200)[:, None]
        kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(vals[:, None])
        log_dens = kde.score_samples(val_grid)
        dens = np.exp(log_dens)
        if scaled:
            #rescales the KDE so that the area under the curve is 1 by diving the  density
            #values by their total area under the density curve (computed with numerical 
            #integration using trapezoid. Then multiplies the normalized density by the 
            #number of observations in that group. That way, bigger groups have proportionally
            #larger area under the curve
            dens = (dens / np.trapz(dens, val_grid[:, 0])) * len(v)
        return dens.max()
            
    global_max_a = data.groupby([gr, "modalityID"], observed=False)[val].apply(lambda v: _scaling(v, False, bandwidth)).max()
    global_max_c = data.groupby([gr, "modalityID"], observed=False)[val].apply(lambda v: _scaling(v, True, bandwidth)).max()

    scale_factor = (width / n_hue / 2)
    
    # now a single loop to plot
    for (groupID, modeID), df_val in data.groupby([gr, "modalityID"], observed=False):
        vals = df_val[val].values
        cols = df_val['rug_colorid'].values
        #can happen when grouping lead to small groups (in which case no modality is looked for)
        #that there is not enough data point for a density shape
        nuniqval = len(set(vals))
        if nuniqval>=clusMinN:
            bw = compute_bandwidth(vals[:, None], bandwidth)
            vmin = vals.min() - cut * bw if cutmin is None else max(cutmin, vals.min() - cut * bw)
            vmax = vals.max() + cut * bw if cutmax is None else min(cutmax, vals.max() + cut * bw)
            val_grid = np.linspace(vmin, vmax, 200)[:, None]
            kde = KernelDensity(kernel=kernel, bandwidth=bw).fit(vals[:, None])
            log_dens = kde.score_samples(val_grid)
            density = np.exp(log_dens)
            val_grid = val_grid[:, 0]  # flatten to 1D for plotting
            density = density / density.max()  # normalize
            # area/count adjustments here
            integral = np.trapz(density, val_grid)
            if density_norm == "area" and integral > 0:
                density = density / integral / global_max_a
            elif density_norm == "count" and integral > 0:
                density = density * len(vals) / integral / global_max_c
            density = density * scale_factor
            
        gr_center = dico_modeID_positions[modeID]
        signleft, signright, signmiddle = dico_modeID_signs[modeID]

        # Density colour
        if density_color == "black":
            density_c = "black"
        elif density_color == "hue":
            density_c = hue_color_map[df_val[hue].unique()[0]]
        elif density_color == "mode":
            density_c = mode_color_map[modeID]
        else:
            density_c = density_color  # a validated color string
    
        # Boxplot colour
        if not showbox:
            box_c = None  
        elif box_color == "black":
            box_c = "black"
        elif box_color == "hue":
            box_c = hue_color_map[df_val[hue].unique()[0]]
        elif box_color == "mode":
            # darker shade if density_color matches, else normal mode color
            base_c = mode_color_map[modeID]
            box_c = darker(base_c) if density_c == base_c else base_c
        else:
            box_c = box_color  # a validated color string

        # jittered positions for the rugplot
        if jitter:
            np.random.seed(random_state) 
            val_positions = vals + np.random.uniform(-0.05*width, 0.05*width, len(vals))
        else:
            val_positions = vals

        # Plot density, boxplot, jittered rug plot
        if split:
            density = density*2
        if orient=='v':
            if nuniqval < clusMinN:
                ax.scatter(np.repeat(gr_center, len(vals)), vals, color=density_c, alpha=density_alpha, s=20)
            else:
                if showdensity:
                    ax.fill_betweenx(val_grid,  gr_center + signleft * density,  gr_center + signright * density, 
                                     color=density_c, alpha=density_alpha)
                    fill = ax.fill_betweenx(val_grid, gr_center + signleft * density, gr_center + signright * density, 
                                            facecolor="none", edgecolor=density_c, alpha=1, linewidth=density_borderlinewidth)  
                if showrug:
                    # Precompute densities for all positions at once
                    li_dens_here = np.interp(val_positions, val_grid, density, left=0, right=0)
                    rug_half = rug_length / (2 * n_hue)
                    xinit = gr_center + signmiddle * width / 5 / n_hue
                    x0 = xinit - rug_half
                    x1 = xinit + rug_half
                    # prepare data arrays
                    li_segments_inside = []
                    li_segments_outside = []
                    li_colors_inside = []
                    li_colors_outside = []
                    for v, dens_here, col in zip(val_positions, li_dens_here, cols):
                        if (xinit - dens_here) <= x0 + 0.01 or split or not showdensity:
                            li_segments_inside.append([[x0, v], [x1, v]])
                            li_colors_inside.append(col)
                        else:
                            rc = rug_outer_color if rug_outer_color is not None else col
                            li_segments_inside.append([[xinit - dens_here, v], [xinit + dens_here, v]])
                            li_colors_inside.append(col)
                            li_segments_outside.extend([[[x0, v], [xinit - dens_here, v]],
                                                        [[xinit + dens_here, v], [x1, v]]])
                            li_colors_outside.extend([rc, rc])
                    if li_segments_inside:
                        lc_in = LineCollection(li_segments_inside, colors=li_colors_inside,
                                               linewidths=rug_linewidths, alpha=1, zorder=2)
                        ax.add_collection(lc_in)
                    if li_segments_outside:
                        lc_out = LineCollection(li_segments_outside, colors=li_colors_outside,
                                                linewidths=rug_outer_linewidths, alpha=1, zorder=2)
                        ax.add_collection(lc_out)
                    
                if showbox:
                    vis = False
                    if not showrug:
                        vis = True
                    ax.boxplot(vals, positions=[gr_center + signmiddle * width/5/n_hue], widths=width/4/n_hue*box_width, patch_artist=True, 
                               boxprops=dict(facecolor='none', edgecolor=box_c, linewidth=box_linewidth),
                               medianprops=dict(color=box_c, linewidth=box_linewidth), 
                               whiskerprops=dict(color=box_c, linewidth=box_linewidth), 
                               capprops=dict(visible=vis, color=box_c), flierprops=dict(marker=''))

        else:
            if nuniqval < clusMinN:
                ax.scatter(vals, np.repeat(gr_center, len(vals)), color=density_c, alpha=density_alpha, s=20)
            else:
                if showdensity:
                    ax.fill_between(val_grid,  gr_center + signleft * density,  gr_center + signright * density, 
                                    alpha=density_alpha, color=density_c)
                    fill = ax.fill_between(val_grid,  gr_center + signleft * density,  gr_center + signright * density, 
                                facecolor="none", alpha=1, edgecolor=density_c, linewidth=density_borderlinewidth)
                if showrug:
                    # Precompute densities for all positions at once
                    li_dens_here = np.interp(val_positions, val_grid, density, left=0, right=0)
                    rug_half = rug_length / (2 * n_hue)
                    yinit = gr_center + signmiddle * width / 5 / n_hue
                    y0 = yinit - rug_half
                    y1 = yinit + rug_half
                    # Prepare containers for line segments
                    li_segments_inside = []
                    li_segments_outside = []
                    li_colors_inside = []
                    li_colors_outside = []
                    for v, dens_here, col in zip(val_positions, li_dens_here, cols):
                        if (yinit - dens_here) <= y0 + 0.01 or split or not showdensity:
                            li_segments_inside.append([[v, y0], [v, y1]])
                            li_colors_inside.append(col)
                        else:
                            rc = rug_outer_color if rug_outer_color is not None else col
                            # Inside (density core)
                            li_segments_inside.append([[v, yinit - dens_here], [v, yinit + dens_here]])
                            li_colors_inside.append(col)
                            # Outside (extensions)
                            li_segments_outside.extend([
                                [[v, y0], [v, yinit - dens_here]],
                                [[v, yinit + dens_here], [v, y1]]
                            ])
                            li_colors_outside.extend([rc, rc])
                    # Add all line collections at once
                    if li_segments_inside:
                        lc_in = LineCollection(li_segments_inside, colors=li_colors_inside,
                                               linewidths=rug_linewidths, alpha=1, zorder=2)
                        ax.add_collection(lc_in)
                    if li_segments_outside:
                        lc_out = LineCollection(li_segments_outside, colors=li_colors_outside,
                                                linewidths=rug_outer_linewidths, alpha=1, zorder=2)
                        ax.add_collection(lc_out)


                if showbox:
                    vis = False
                    if not showrug:
                        vis = True
                    ax.boxplot(vals, positions=[gr_center + signmiddle * width/5/n_hue], widths=width/4/n_hue*box_width, vert=False,
                               patch_artist=True, 
                               boxprops=dict(facecolor='none', edgecolor=box_c, linewidth=box_linewidth),
                               medianprops=dict(color=box_c, linewidth=box_linewidth), 
                               whiskerprops=dict(color=box_c, linewidth=box_linewidth),
                               capprops=dict(visible=vis, color=box_c), flierprops=dict(marker=''))                 
            # Center y-axis labels on group centers
            ax.set_yticks([gr_center])  # ensure tick exactly matches group center
            ax.set_yticklabels([str(groupID)], va='center')



    # ------------------------------------------------------------------ #
    # Add legend
    # ------------------------------------------------------------------ #
    legend_handles = []
    legend_labels = []
    
    # Legend - modality
    if mode_legend:
        mode_handles = [Patch(facecolor=color, edgecolor="black", label=modeID)
                        for modeID, color in mode_color_map.items()]
        legend_handles.extend(mode_handles)
        legend_labels.extend([h.get_label() for h in mode_handles])
    
    # Legend - hue
    if hue is not None and hue_legend:
        hue_handles = [Patch(facecolor=color, edgecolor="black", label=hue_val)
                       for hue_val, color in hue_color_map.items()]
        legend_handles.extend(hue_handles)
        legend_labels.extend([h.get_label() for h in hue_handles])
    
    # Legend - rug
    if rug_hue is not None and rug_legend:
        if pd.api.types.is_numeric_dtype(data[rug_hue]):
            sm = mpl.cm.ScalarMappable(cmap=rug_palette, norm=rug_norm)
            if rug_colorbarposition is not None:
                try:
                    cbar_ax = ax.figure.add_axes(rug_colorbarposition)
                    plt.colorbar(sm, cax=cbar_ax, label=rug_hue,
                                 aspect=rug_colorbarthickness, shrink=rug_colorbarheight)
                except Exception as e:
                    warnings.warn(
                        "Invalid rug_colorbarposition argument, it should be [cbar_x, cbar_y, cbar_width, cbar_height]."
                        f" Error: {e}", stacklevel=2)
                    plt.colorbar(sm, ax=ax, label=rug_hue,
                                 aspect=rug_colorbarthickness, shrink=rug_colorbarheight)
            else:
                plt.colorbar(sm, ax=ax, label=rug_hue,
                             aspect=rug_colorbarthickness, shrink=rug_colorbarheight)
        else:
            rug_handles = [Line2D([0], [0], marker='_', color=col, linestyle='None',
                                  markersize=10, markeredgewidth=2, label=val)
                           for val, col in data.drop_duplicates(rug_hue)[[rug_hue, 'rug_colorid']].values]
            legend_handles.extend(rug_handles)
            legend_labels.extend([h.get_label() for h in rug_handles])
    
    # Final combined legend 
    if legend_handles:
        try:
            legend_kwargs = dict(frameon=True, handles=legend_handles, labels=legend_labels)
    
            # --- Add a title ONLY if there's one legend type
            n_legends = sum([
                bool(mode_legend),
                bool(hue is not None and hue_legend),
                bool(rug_hue is not None and rug_legend)
            ])
            legend_title = None
            if n_legends == 1:
                if mode_legend:
                    legend_title = "Modality"
                elif hue is not None and hue_legend:
                    legend_title = hue
                elif rug_hue is not None and rug_legend:
                    legend_title = rug_hue
    
            if legend_position is None:
                legend_obj = ax.legend(loc="upper left", title=legend_title, **legend_kwargs)
            elif isinstance(legend_position, (list, tuple)):
                legend_obj = ax.legend(bbox_to_anchor=legend_position, title=legend_title, **legend_kwargs)
            else:
                legend_obj = ax.legend(loc=legend_position, title=legend_title, **legend_kwargs)
    
            if legend_title is not None:
                legend_obj.get_title().set_fontsize(10)
    
        except Exception:
            warnings.warn(
                "Invalid `legend_position`: must be None, a valid string location (e.g., 'upper right'), "
                "or a tuple/list of coordinates. Using default 'upper left'.", stacklevel=2
            )
            ax.legend(loc="upper left", **legend_kwargs)
    
        
    if orient=='v':
        ax.set_xticks(list(dico_groupID_positions.values()))
        ax.set_xticklabels(list(dico_groupID_positions.keys()))
        ax.set_xlim(-0.5, len(list(dico_groupID_positions.keys())) - 0.5)                
    else:
        ax.set_yticks(list(dico_groupID_positions.values()))
        ax.set_yticklabels(list(dico_groupID_positions.keys()))
        ax.set_ylim(-0.5, len(list(dico_groupID_positions.keys())) - 0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    # Merge back with nan
    result = pd.merge(init_data, data[['modalityID', 'unique_idbixplot']], on='unique_idbixplot', how="left")
    # Drop helper column
    data = data.drop(columns=['rug_colorid', 'unique_idbixplot'])

    return dico_xhue_results, result


def bixplot_methods(data=None, group_vars=None, kde_var=None, 
                    kmax=7, minN=15, clusMinN=3, stand=False, maxit=500, diplevel=0.01, 
                    verbose=False, undo_constrainkmax=False, random_state=0):

    """
    Test for unimodality within groups of data and assign modality IDs.

    Parameters
    ----------
    data : DataFrame, required
        Input dataset containing the variable of interest (`kde_var`) and grouping columns (group_vars).
    group_vars : list of str, required
        Column names used to define groups that should be analyzed independently.
    kde_var : str, required
        Column name for the numeric variable to analyze.
    minN : int, default = 15
        the minimum number of observations required
        per potential cluster. The maximum number of 
        clusters searched is limited to n/minN, where n is
        the number of non-missing values. If n < 2 ∗ minN, 
        clustering is not attempted to avoid spurious
        clusters from small samples. 
    clusMinN : int, default = 3
        The clustering is constrained so that any cluster must contain at 
        least clusMinN unique values. When a variable has fewer values 
        than this, only points are drawn.
    kmax : int, default=None    
        Maximum number of clusters to test. It will be truncated to 5 
        internally. If NULL, it is set to min(floor(n/minN), 5). 
        When setting kmax=1 all variables are considered as single 
        clusters, making the display resemble a violin plot.
    maxit : int, default=500
        Maximum number of iterations in the constrained clustering loop.
    verbose : bool, defualt = False
        If True, will print intermediate steps
    random_state : int or None, default=0
        Random seed passed to mixture or clustering algorithms.
        Ensures reproducibility when set.

    Returns
    -------
    dico_xhue_results : dict
        Dictionary keyed by group ID string (one entry for each unique combination of values in group_vars).
        Each entry contains:
        - 'default_unimodal_due_to_sample' : bool
        - 'dip_test_statistic', 'dip_test_p_value'
        - 'gmm_best_k' / 'kmedoids_best_k'
        - 'modalityID' : array of cluster assignments
        - model parameters (means, stds, weights or silhouette scores).
    df_withmodalitylabels : DataFrame
        A copy of `data` with an added column:
        - 'modalityID' : str labels of form "<group_id>_<cluster_id>".

    Notes
    -----
    - The Hartigan dip test is used at 1% significance level to decide
      whether to test for multimodality.
    - Cluster IDs are remapped to consecutive integers (0, 1, 2, …) to ensure consistency across groups.
    - If sample size or unique values are insufficient, unimodality is assumed.
    """
    
    dico_xhue_results = {}
    li_df = []
    solver = pulp.PULP_CBC_CMD(msg=False)

    for keys, df_ in data.groupby(group_vars, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_id = "_".join(str(k) for k in keys)

        df_ = df_.copy()
        df_['modalityID'] = group_id+ '_0'

        #check appropriate sample size 
        values = df_[kde_var].to_numpy()
        n = len(values)
        nuniq = len(set(values))
        
        ######### Determine kmax
        mykmax = kmax or max(1, math.floor(n/minN))
        if not undo_constrainkmax:
            mykmax = min(mykmax, 7)
        mykmax = max(1, min(mykmax, math.floor(nuniq/clusMinN)))
        if mykmax * minN > n:
            mykmax = max(1, math.floor(n/minN))
                
        results = {'default_unimodal_due_to_sample': False}
        results = {'kmax': mykmax}
        
        if mykmax==1:
            results['default_unimodal_due_to_sample'] = True
        else:
            ######### Determine modes
            dip_stat, p_dip = diptest(values)
            results.update({'dip_test_statistic': dip_stat, 'dip_test_p_value': p_dip})
            if p_dip <= diplevel:
                best_score = -1
                best_k = 1
                best_labels = np.zeros_like(values)
                #when kmax=1 then it skips the loop
                for k in range(2, mykmax+1):
                    pam_result = pamc1d(y=values, k=k, solver=solver, minsize=clusMinN, countwhat="unique", stand=stand, 
                                        maxit=maxit, verbose=verbose, random_state=random_state)
                    labels_ = pam_result["clustering"]
                    if len(np.unique(labels_)) > 1:  # only compute if at least 2 clusters
                        score = silhouette_score(values.reshape(-1, 1), labels_)
                        #score = silhouette_score(X, labels_)
                    else:
                        score = -1                                         
                    # Only update if the score is strictly better
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels_
                results.update({'best_k': best_k, 'silhouette_score': best_score, 'modalityID': best_labels})

                # Remap cluster labels so that _0 has the lowest mean, _1 the second lowest, etc.
                cluster_means = {label: values[best_labels == label].mean() for label in np.unique(best_labels)}
                sorted_labels = sorted(cluster_means, key=cluster_means.get, reverse=False)
                label_map = {old: new for new, old in enumerate(sorted_labels)}
                best_labels = np.array([label_map[l] for l in best_labels])
                df_['modalityID'] = group_id + "_" + pd.Series(best_labels, index=df_.index).astype(str)

        dico_xhue_results[group_id] = results
        li_df.append(df_)
        
    return dico_xhue_results, pd.concat(li_df)
   