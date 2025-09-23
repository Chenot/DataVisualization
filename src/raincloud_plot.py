# Raincloud Plot Function
# Author: Quentin Chenot
# License: MIT
# Description: A flexible function for creating raincloud plots for 1-way or 2-way designs.
# Version: 1.0
# Date: 12/09/2025 [DD/MM/YYYY]

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from itertools import product
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import gaussian_kde
from typing import Optional, List, Tuple

def plot_raincloud(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: Optional[str] = None,
    point_alpha: float = 0.7,
    point_size: int = 25,
    boxplot_alpha: float = 0.8,
    violin_alpha: float = 0.8,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    anova_pos: str = 'top-right',
    colors: Optional[List[str]] = None,
    x1_order: Optional[List[str]] = None,
    x2_order: Optional[List[str]] = None
) -> None:
    """
    Create raincloud plots (half-violin + boxplot + jitter) with optional ANOVA.

    Parameters (short)
    - data: pd.DataFrame containing columns named by y, x1 (and x2 if used).
    - y: dependent numeric column name.
    - x1: primary categorical factor column name.
    - x2: secondary categorical factor column name (optional).
    - point_alpha: transparency for jittered points (0-1).
    - point_size: marker size for points.
    - boxplot_alpha: box fill transparency (0-1).
    - violin_alpha: violin fill transparency (0-1).
    - box_color: fallback color for boxplots.
    - x_label, y_label, title: axis labels and plot title (optional).
    - figsize: figure size in inches as (width, height).
    - save_path: file path to save plot (creates directories if needed).
    - anova_pos: placement of ANOVA text, e.g. 'top-right' or 'bottom-left'.
    - colors: list of colors (length must match number of groups).
    - x1_order, x2_order: explicit ordering of factor levels (optional).

    Returns
    - fig, ax: matplotlib Figure and Axes for further customization.

    Notes
    - Groups are formed as "x1" or "x1 x2". ANOVA is computed with statsmodels OLS
      using categorical encoding C(...); interaction included if x2 is provided.
    """

    df = data.copy()

    # Apply custom order for x1 and x2 if provided
    if x1_order:
        df[x1] = pd.Categorical(df[x1], categories=x1_order, ordered=True)
    if x2 and x2_order:
        df[x2] = pd.Categorical(df[x2], categories=x2_order, ordered=True)

    # Create grouping column
    if x2:
        df['group'] = df[x1].astype(str) + " " + df[x2].astype(str)  # Use " " as a delimiter
    else:
        df['group'] = df[x1].astype(str)

    # Sort groups based on x1 and x2 orders
    groups = sorted(df['group'].unique(), key=lambda g: (
        x1_order.index(g.split(" ")[0]) if x1_order else g.split(" ")[0],  # Split by " "
        x2_order.index(" ".join(g.split(" ")[1:])) if x2 and x2_order else " ".join(g.split(" ")[1:]) if x2 else ""
    ))
    positions = np.arange(len(groups))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Assign colors
    n_groups = len(groups)
    if colors:
        if len(colors) != n_groups:
            raise ValueError(f"Number of colors ({len(colors)}) must match the number of groups ({n_groups}).")
        palette = colors  # Use user-provided colors
    else:
        palette = sns.color_palette("RdBu", n_groups)  # Default palette

    group_colors = dict(zip(groups, palette))  # Map each group to a color

    # Raincloud components
    for i, grp in enumerate(groups):
        vals = df.loc[df['group'] == grp, y].values
        color = group_colors[grp]  # Get the color for the current group

        # --- Half violin (right side) ---
        kde = gaussian_kde(vals)
        y_grid = np.linspace(vals.min(), vals.max(), 200)
        dens = kde(y_grid)

        # Normalize density and scale width
        dens = dens / dens.max() * 0.3  

        # Shift density to the right of group position
        ax.fill_betweenx(y_grid, i, i + dens,
                        facecolor=color, alpha=violin_alpha, zorder=1)

        # Add black contour to the density
        ax.plot(i + dens, y_grid, color='black', linewidth=0.8, zorder=2)  # Black outline

        # --- Boxplot (slightly left) ---
        ax.boxplot(vals, positions=[i - 0.055], widths=0.1,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=boxplot_alpha),
                medianprops=dict(color='black'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', markerfacecolor='black', alpha=0))

        # --- Jittered scatter (left) ---
        x_jitter = np.random.uniform(low=i - 0.27, high=i - 0.12, size=len(vals))
        ax.scatter(x_jitter, vals,
                color=color, alpha=point_alpha,
                s=point_size + 2,  # Slightly larger dots
                edgecolor='black', linewidth=0.5, zorder=10)  # Add black contour

    # ANOVA
    anova_text = ""
    try:
        if x2:
            formula = f"{y} ~ C({x1}) * C({x2})"
        else:
            formula = f"{y} ~ C({x1})"
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        lines = []
        # Update ANOVA results formatting
        for factor in anova_table.index:
            clean_factor = factor.replace("C(", "").replace(")", "").replace(":", "*")
            p = anova_table.loc[factor, 'PR(>F)']
            if pd.isna(p):
                continue
            if p < 0.001:
                p_text = "p < .001"
            else:
                p_text = f"p = {p:.3f}".replace("0.", ".")
            lines.append(f"{clean_factor}: {p_text}")
        anova_text = "\n".join(lines)
    except Exception:
        anova_text = "ANOVA failed"

    # Axis labels and title
    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)
    if title: ax.set_title(title)

    # ANOVA annotation placement
    y_max = df[y].max()
    y_min = df[y].min()
    x_min, x_max = positions.min(), positions.max()

    # Horizontal placement (± 0.3)
    xpos = (x_min - 0.3) if "left" in anova_pos else (x_max + 0.3)

    # Vertical placement (above or below data range)
    ypos = (y_max * 1.05) if "top" in anova_pos else (y_min * 0.95)

    ha = "left" if "left" in anova_pos else "right"
    va = "bottom" if "top" in anova_pos else "top"

    ax.text(xpos, ypos, anova_text,
            ha=ha, va=va, fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"))  # Removed black line

    # --- Expand axes so annotation is visible ---
    ax.set_xlim(x_min - 0.6, x_max + 0.6)
    ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max * 1.2)

    plt.tight_layout()

    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=600, bbox_inches="tight")

    plt.show()

    return fig, ax  # Return the figure and axis objects


def generate_1way_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic one-way design data (dataframe with condition and score column)
    """
    np.random.seed(seed)
    conditions = np.random.choice(["Control", "Experimental"], size=n_samples)

    scores = []
    for condition in conditions:
        if condition == "Control":
            scores.append(np.random.normal(loc=20, scale=5))
        else:  # Experimental
            scores.append(np.random.normal(loc=25, scale=5))

    return pd.DataFrame({
        "Condition": conditions,
        "Score": scores
    })


def generate_2way_data(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic two-way design data (dataframe with factor1, factor2 and score column)
    """
    np.random.seed(seed)
    factor1_levels = ["Low", "High"]
    factor2_levels = ["Type A", "Type B"]
    conds = list(product(factor1_levels, factor2_levels))
    n_conditions = len(conds)

    assignments = [conds[i % n_conditions] for i in range(n_samples)]
    scores = []

    for factor1, factor2 in assignments:
        if factor1 == "Low" and factor2 == "Type A":
            scores.append(np.random.normal(loc=15, scale=4))
        elif factor1 == "Low" and factor2 == "Type B":
            scores.append(np.random.normal(loc=18, scale=4))
        elif factor1 == "High" and factor2 == "Type A":
            scores.append(np.random.normal(loc=22, scale=4))
        else:  # High + Type B
            scores.append(np.random.normal(loc=15, scale=4))

    return pd.DataFrame({
        "Factor1": [a[0] for a in assignments],
        "Factor2": [a[1] for a in assignments],
        "Score": scores
    })


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Example 1: One-way design
    df1 = generate_1way_data()
    plot_raincloud(df1, y="Score", x1="Condition",
                   x_label="Condition", y_label="Score",
                   figsize=(10, 6),
                   save_path=os.path.join(base_dir, "figures", "raincloud_plot_1way.pdf"))

    # Example 2: Two-way design
    df2 = generate_2way_data()
    plot_raincloud(df2, y="Score", x1="Factor1", x2="Factor2",
                   y_label="Score",
                   figsize=(12, 6),
                   x1_order=["Low", "High"],
                   x2_order=["Type A", "Type B"],
                   colors=["#f4a582", "#67001f", "#92c5de", "#053061"],
                   save_path=os.path.join(base_dir, "figures", "raincloud_plot_2way.pdf"))


