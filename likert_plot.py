# Likert Plot Function
# Author: Quentin Chenot
# License: MIT
# Description: A flexible function for creating data visualization from questionnaires using Likert scales.
# Version: 1.0
# Date: 15/09/2025 [DD/MM/YYYY]

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from typing import Optional, List, Tuple

def plot_likert(data: pd.DataFrame,
                question: str,
                categories: str,
                colors: Optional[List[str]] = None,
                cmap: str = 'coolwarm',
                cmap_reverse: bool = True,
                figsize: Tuple[int, int] = (8, 6),
                category_order: Optional[List[str]] = None,
                label_rotation: int = 0,
                legend_title: str = "Response",
                bar_height: float = 0.6,
                save_path: Optional[str] = None,
                save_percents_path: Optional[str] = None,
                sort_questions: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw a diverging (Likert) stacked horizontal bar chart with the neutral category centered.

    Parameters (short)
    - data: long-format pd.DataFrame (line: participants; row: questions).
    - question: column name with question/item labels (y-axis).
    - categories: column name with response category labels.
    - colors: list of colors matching category_order (overrides cmap).
    - cmap: matplotlib colormap name used when colors is None.
    - cmap_reverse: reverse the colormap when True.
    - figsize: figure size (width, height) in inches.
    - category_order: explicit left-to-right category order; inferred if None.
    - label_rotation: rotation angle for x tick labels (degrees).
    - legend_title: title for the legend.
    - bar_height: thickness of the horizontal bars.
    - save_path: file path to save figure (dirs created).
    - save_percents_path: if provided, save per-question percent table to CSV.
    - sort_questions: sort question labels alphabetically when True.

    Returns
    - fig, ax: matplotlib Figure and Axes.

    Notes
    - Colors may be explicit or derived from cmap.
    - Color maps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    """

    df = data.copy()

    # If category_order not provided, infer stable order from data
    if category_order is None:
        # ensure string type and stable ordering of categories
        category_order = list(df[categories].astype(str).unique())

    # enforce categories as ordered categorical to avoid dtype-mixing issues with plotting
    df[categories] = pd.Categorical(df[categories].astype(str),
                                    categories=category_order,
                                    ordered=True)

    # compute counts and percentages per question (rows sum to 100)
    counts = df.groupby([question, categories], observed=False).size().unstack(fill_value=0)
    counts = counts.reindex(columns=category_order, fill_value=0)
    percents = counts.div(counts.sum(axis=1), axis=0) * 100

    # optionally save the percent table for reproducibility / publication supplement
    if save_percents_path:
        os.makedirs(os.path.dirname(save_percents_path) or ".", exist_ok=True)
        percents.to_csv(save_percents_path)

    # colors: prefer explicit list, otherwise build from cmap
    if colors is None:
        # get colormap object robustly
        try:
            cmap_obj = plt.colormaps[cmap]
        except Exception:
            cmap_obj = plt.get_cmap(cmap)
        if cmap_reverse:
            # use reversed colormap
            try:
                cmap_obj = cmap_obj.reversed()
            except Exception:
                cmap_obj = plt.get_cmap(cmap + "_r")
        colors = [cmap_obj(i) for i in np.linspace(0, 1, len(category_order))]

    # split categories into left / neutral / right
    mid = len(category_order) // 2
    has_neutral = (len(category_order) % 2 == 1)
    if has_neutral:
        left_cats = category_order[:mid]
        neutral_cat = category_order[mid]
        right_cats = category_order[mid + 1:]
    else:
        left_cats = category_order[:mid]
        neutral_cat = None
        right_cats = category_order[mid:]

    # optionally sort questions (useful for reproducible figures)
    if sort_questions:
        percents = percents.sort_index()

    # prepare plot
    fig, ax = plt.subplots(figsize=figsize)
    y_positions = np.arange(len(percents.index))

    # compute left positions so that neutral is exactly centered around 0:
    # neutral_left = - neutral_pct/2, neutral_right = + neutral_pct/2
    left_positions = pd.DataFrame(0.0, index=percents.index, columns=percents.columns)

    for idx in percents.index:
        row = percents.loc[idx]
        neutral_width = row[neutral_cat] if neutral_cat is not None else 0.0
        neutral_left = - neutral_width / 2.0
        if neutral_cat is not None:
            left_positions.at[idx, neutral_cat] = neutral_left

        # stack right categories from neutral_right outward
        right_start = neutral_width / 2.0
        for cat in right_cats:
            left_positions.at[idx, cat] = right_start
            right_start += row[cat]

        # stack left categories leftwards from neutral_left outward
        left_start = neutral_left
        for cat in left_cats[::-1]:
            left_start -= row[cat]
            left_positions.at[idx, cat] = left_start

    # draw bars using positive widths and computed left positions (no negative widths)
    for cat in category_order:
        widths = percents[cat].values
        lefts = left_positions[cat].values
        ax.barh(y_positions, widths, left=lefts,
                color=colors[category_order.index(cat)],
                edgecolor='black', height=bar_height, label=cat)

    # formatting: y labels are question names
    ax.set_yticks(y_positions)
    ax.set_yticklabels(percents.index)
    ax.set_xlabel("% of responses")
    ax.set_xlim(-100, 100)
    ax.axvline(0, color='black', linewidth=0.8, zorder=0)
    ax.set_xticks(np.linspace(-100, 100, 11))

    def _text_contrast_color(rgb):
        # rgb in [0,1] tuple -> compute luminance, return 'white' or 'black'
        r, g, b = rgb[:3]
        # relative luminance
        lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return 'black' if lum > 0.5 else 'white'

    # optional: add percent labels inside bars when width > threshold
    for i, idx in enumerate(percents.index):
        for cat in category_order:
            w = percents.at[idx, cat]
            if w <= 0.5:
                continue
            lx = left_positions.at[idx, cat]
            cx = lx + w / 2.0
            text_color = _text_contrast_color(colors[category_order.index(cat)])
            ax.text(cx, i, f"{w:.0f}%", ha='center', va='center', color=text_color, fontsize=8)

    # legend and layout
    # place legend centered below the x-axis label and keep a reference so it can be included in saved output
    ncol = len(category_order)
    legend = ax.legend(title=legend_title,
                       loc='upper center',
                       bbox_to_anchor=(0.5, -0.12),
                       ncol=ncol,
                       frameon=False)
    plt.xticks(rotation=label_rotation)
    plt.tight_layout()
    # reserve space for the legend below the axes
    plt.subplots_adjust(bottom=0.18)

    # save outputs: figure and optional percents already handled
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            plt.savefig(save_path, dpi=600, bbox_inches='tight', bbox_extra_artists=(legend,))
        except Exception:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()
    return fig, ax


# Example usage: convert your wide table (questions in columns) to long format and call plot_likert
if __name__ == "__main__":
    df = pd.DataFrame({
        'Participant': [1, 2, 3, 4, 5],
        'We are all': [5, 4, 3, 2, 1],
        'We mostly disagree': [1, 2, 2, 2, 3],
        'We mostly agree': [3, 4, 4, 4, 5],
        'We all strongly disagree': [1, 1, 1, 1, 1],
        'We all  strongly agree': [5, 5, 5, 5, 5]
    })

    # melt to long format (questions as rows)
    df_long = df.melt(id_vars=['Participant'],
                      var_name='Question',
                      value_name='Response')

    # map numeric to labels if needed
    response_mapping = {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neutral",
        4: "Agree",
        5: "Strongly agree"
    }
    df_long['Response'] = df_long['Response'].map(response_mapping)

    # Example: reversed coolwarm (default), also export percents for reproducibility
    plot_likert(df_long,
                question='Question',
                categories='Response',
                category_order=["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
                cmap='coolwarm',
                cmap_reverse=True,
                save_path="output/likert_plot.pdf",
                figsize=(10, 6))