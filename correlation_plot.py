# Correlation Plot Function
# Author: Quentin Chenot
# License: MIT
# Description: A flexible function for creating scatter plots with correlation and density.
# Version: 1.1
# Date: 15/09/2025 [DD/MM/YYYY]
 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import os
from typing import Optional, Tuple

def plot_correlation(
    data: pd.DataFrame,
    x: str,
    y: str,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    show_density: bool = True,
    point_color: str = 'black',
    point_alpha: float = 0.7,
    point_size: int = 30,
    point_marker: str = 'o',
    line_color: str = 'black',
    line_style: str = '-',
    line_ci: Optional[float] = 95,
    cor_pos: str = 'top-right',
    cor_fontsize: int = 12,
    cor_color: str = 'black',
    cor_bbox_alpha: float = 0.5,
    margin_ratio: float = 0.02,
    figsize: Tuple[int, int] = (6, 6),
    x_scale: str = 'linear',
    y_scale: str = 'linear',
    density_color: str = 'grey',
    density_alpha: float = 0.5,
    save_path: Optional[str] = None,
    corr_method: str = 'pearson'
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Scatterplot with optional regression, marginal densities and correlation annotation.

    Parameters (short)
    - data: pd.DataFrame containing columns named by x and y.
    - x, y: column names for x and y axes (numeric).
    - x_label, y_label: axis labels (defaults to column names).
    - show_density: add marginal density plots when True.
    - point_color, point_alpha, point_size, point_marker: marker styling.
    - line_color, line_style: regression line styling.
    - line_ci: confidence interval for regression (percent) or None to disable.
    - cor_pos: location for correlation text ('top-right', 'top-left', 'bottom-right', 'bottom-left').
    - cor_fontsize, cor_color, cor_bbox_alpha: correlation text style and background alpha.
    - margin_ratio: extra axis margin as fraction of data range.
    - figsize: figure size (width, height) in inches.
    - x_scale, y_scale: 'linear' or 'log' axis scales.
    - density_color, density_alpha: marginal density styling.
    - save_path: file path to save figure (if provided).
    - corr_method: 'pearson', 'spearman' or 'kendall' for correlation computation.

    Returns
    - fig, ax: matplotlib Figure and Axes.

    Notes
    - Automatically computes and displays correlation coefficient and p-value using corr_method.
    """
    # Compute correlation
    # Prepare data and drop NaNs
    df = data[[x, y]].dropna()
    if df.empty:
        raise ValueError("No valid (non-NaN) observations found for the specified x and y.")

    # Handle constant columns (pearson/spearman undefined)
    if df[x].nunique() < 2 or df[y].nunique() < 2:
        r = float('nan')
        p_value = float('nan')
    else:
        if corr_method == 'pearson':
            r, p_value = stats.pearsonr(df[x], df[y])
        elif corr_method == 'spearman':
            r, p_value = stats.spearmanr(df[x], df[y])
        else:
            raise ValueError("corr_method must be 'pearson' or 'spearman'")

    r2 = r**2 if pd.notna(r) else float('nan')

    # American format (no leading zero)
    def fmt(val):
        if abs(val) < 1:
            return f"{val:.3f}".replace("0.", ".")
        else:
            return f"{val:.3f}"

    p_text = "p < .001" if p_value < 0.001 else f"p = {fmt(p_value)}"
    cor_text = f"r = {fmt(r)}, r² = {fmt(r2)}, {p_text}"

    # Margins
    x_min, x_max = data[x].min(), data[x].max()
    y_min, y_max = data[y].min(), data[y].max()
    x_margin = (x_max - x_min) * margin_ratio
    y_margin = (y_max - y_min) * margin_ratio
    x_limits = (x_min - x_margin, x_max + x_margin)
    y_limits = (y_min - y_margin, y_max + y_margin)

    # Create plot
    if show_density:
        g = sns.JointGrid(data=df, x=x, y=y, height=figsize[1], ratio=4, space=0.2)
        g.plot_joint(sns.scatterplot, color=point_color, alpha=point_alpha,
                     s=point_size, marker=point_marker)
        if pd.notna(r):  # draw regression only when defined
            g.plot_joint(sns.regplot, scatter=False,
                         line_kws={'color': line_color, 'linestyle': line_style},
                         ci=line_ci)
        # draw filled marginal KDE (fill color) without its contour line, then overlay a black contour
        g.plot_marginals(sns.kdeplot, fill=True, color=density_color, alpha=density_alpha, linewidth=0)
        g.plot_marginals(sns.kdeplot, fill=False, color='black', linewidth=1)
        ax = g.ax_joint
        fig = g.fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.scatterplot(data=df, x=x, y=y, color=point_color, alpha=point_alpha,
                        s=point_size, marker=point_marker, ax=ax)
        if pd.notna(r):
            sns.regplot(data=df, x=x, y=y, scatter=False, color=line_color,
                        line_kws={'linestyle': line_style}, ci=line_ci, ax=ax)

    # Set tight axis limits
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)

    # Set axis scales
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    # Set labels
    if x_label: ax.set_xlabel(x_label)
    if y_label: ax.set_ylabel(y_label)

    # Annotation position
    xpos = {'left': x_limits[0], 'right': x_limits[1]}[cor_pos.split('-')[1]]
    ypos = {'top': y_limits[1], 'bottom': y_limits[0]}[cor_pos.split('-')[0]]

    ax.text(xpos, ypos, cor_text,
            horizontalalignment='left' if 'left' in cor_pos else 'right',
            verticalalignment='bottom' if 'bottom' in cor_pos else 'top',
            fontsize=cor_fontsize,
            color=cor_color,
            bbox=dict(facecolor='white', alpha=cor_bbox_alpha, edgecolor='none'))

    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()
    # Always return (fig, ax) for consistent API
    return fig, ax

# Example usage
if __name__ == "__main__":
    df = sns.load_dataset('mpg').dropna(subset=['horsepower', 'mpg'])
    plot_correlation(df,
                     'horsepower',
                     'mpg',
                     x_label='Horsepower',
                     y_label='Miles per Gallon',
                     show_density=True,
                     cor_pos='top-right',
                     figsize=(8,6),
                     save_path='output/correlation_plot.pdf')
