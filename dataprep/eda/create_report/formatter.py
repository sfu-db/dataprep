"""
    This module implements the formatting
    for create_report(df) function.
"""
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional

import dask
import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.models import Title
from bokeh.plotting import Figure

from ..correlation import compute_correlation, render_correlation
from ..distribution import compute, render
from ..distribution.compute import calc_stats
from ..distribution.render import (
    format_cat_stats,
    format_num_stats,
    format_overview_stats,
)
from ..dtypes import Continuous, DateTime, Nominal, detect_dtype, is_dtype
from ..intermediate import Intermediate
from ..missing import compute_missing, render_missing
from ..utils import is_notebook, to_dask

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def format_report(df: pd.DataFrame, mode: Optional[str]) -> Dict[str, Any]:
    """
    Format the data and figures needed by report

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    mode: Optional[str]
        This controls what type of report to be generated.
        Currently only the 'basic' is fully implemented.

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data will be stored.
        This variable acts like an API in passing data to the template engine.
    """

    comps: Dict[str, Any] = {}
    if mode == "basic":
        comps = format_basic(df, comps)
    # elif mode == "full":
    #     comps = format_full(df, comps)
    # elif mode == "minimal":
    #     comps = format_mini(df, comps)
    else:
        raise ValueError

    return comps


def format_basic(df: pd.DataFrame, comps: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format basic version.

    Parameters
    ----------
    df
        The DataFrame for which data are calculated.
    comp:
        A dictionary in which formatted data will be stored.

    Returns
    -------
    Dict[str, Any]
        A dictionary in which formatted data is stored.
        This variable acts like an API in passing data to the template engine.
    """
    # pylint: disable=too-many-locals, too-many-statements, broad-except, too-many-branches

    num_cols: List[str] = []
    dtype_cnts: DefaultDict[str, int] = defaultdict(int)
    for col in df.columns:
        col_dtype = detect_dtype(df[col])
        if is_dtype(col_dtype, Nominal()):
            dtype_cnts["Categorical"] += 1
        elif is_dtype(col_dtype, Continuous()):
            dtype_cnts["Numerical"] += 1
            num_cols.append(col)
        elif is_dtype(col_dtype, DateTime()):
            dtype_cnts["DateTime"] += 1

    with tqdm(total=7 + 2 * len(df.columns) + len(num_cols) ** 2) as pbar:
        # Missing Values
        itmdt = compute_missing(df)
        pbar.set_description(desc="Computing Missing Values")
        pbar.update(1)
        if any(itmdt["data_total_missing"].values()):
            comps["has_missing"] = True
            try:
                rendered = render_missing(itmdt)
                comps["missing"] = components(
                    [
                        _morph_figure(
                            tab.child.children[0],
                            sizing_mode="stretch_width",
                            title=Title(
                                text=tab.title, align="center", text_font_size="20px"
                            ),
                        )
                        for tab in rendered.tabs
                    ]
                )
                pbar.set_description(desc="Formating Missing Values")
            except Exception as error:
                comps["missing"] = (0, {"error": error})  # same template for rendering
                pbar.set_description(desc="Something Happened...")
        else:
            comps["has_missing"] = False
            pbar.set_description(desc="Skipping Missing Values")
        pbar.update(1)

        # Overview
        stats = dask.compute(calc_stats(to_dask(df), dtype_cnts))[0]
        stats = format_overview_stats(stats)
        pbar.set_description(desc="Computing Overview")
        pbar.update(1)
        comps["overview"] = (stats, dtype_cnts)
        pbar.set_description(desc="Formating Overview")
        pbar.update(1)

        # Variables
        comps["variables"] = {}
        for col in df.columns:
            try:
                itmdt = dask.compute(compute(df, col, top_words=15))[0]
                pbar.set_description(desc=f"Computing {col}")
                pbar.update(1)
                rendered = render(itmdt)
                if is_dtype(detect_dtype(df[col]), Nominal()):
                    tabledata = format_cat_stats(*itmdt["stats"])
                elif is_dtype(detect_dtype(df[col]), Continuous()):
                    tabledata = format_num_stats(itmdt["stats"])
                elif is_dtype(detect_dtype(df[col]), DateTime()):
                    tabledata = itmdt["statsdata"]
                comps["variables"][col] = {
                    "tabledata": tabledata,
                    "plots": components(
                        [
                            _morph_figure(
                                tab.child.children[0]
                                if hasattr(tab.child, "children")
                                else tab.child,
                                plot_width=280,
                                plot_height=250,
                                title=Title(text=tab.title, align="center"),
                            )
                            for tab in rendered.tabs[1:]
                        ]
                    ),  # skip Div
                    "col_type": itmdt.visual_type.replace("_column", ""),
                }
                pbar.set_description(desc=f"Formating {col}")
                pbar.update(1)
            except Exception as error:
                comps["variables"][col] = {"error": error, "plots": (0, 0)}
                pbar.set_description(desc="Something Happened...")
                pbar.update(2)

        # Correlations
        try:
            itmdt = compute_correlation(df)
            pbar.set_description(desc="Computing Correlations")
            pbar.update(1)
            if len(itmdt) != 0:
                comps["has_correlation"] = True
                rendered = render_correlation(itmdt)
                comps["correlations"] = components(
                    [
                        _morph_figure(
                            tab.child,
                            sizing_mode="stretch_width",
                            title=Title(
                                text=tab.title, align="center", text_font_size="20px"
                            ),
                        )
                        for tab in rendered.tabs
                    ]
                )
                pbar.set_description(desc="Formating Correlations")
                pbar.update(1)
            else:
                comps["has_correlation"] = False
                pbar.set_description(desc="Skipping Correlations")
                pbar.update(1)
        except Exception as error:
            comps["has_correlation"] = True
            comps["correlations"] = (0, {"error": error})  # same template for rendering
            pbar.set_description(desc="Something Happened...")
            pbar.update(2)

        # Interactions
        df_coeffs: pd.DataFrame = pd.DataFrame({})
        if len(num_cols) > 1:
            comps["has_interaction"] = True
            # set initial x,y axis value
            df_scatter = df.loc[:, num_cols]
            df_scatter.loc[:, "__x__"] = df_scatter.iloc[:, 0]
            df_scatter.loc[:, "__y__"] = df_scatter.iloc[:, 0]
            try:
                for v_1 in num_cols:
                    for v_2 in num_cols:
                        itmdt = compute_correlation(df, v_1, v_2)
                        coeff_a, coeff_b = itmdt["coeffs"]
                        line_x = np.asarray(
                            [
                                itmdt["data"].iloc[:, 0].min(),
                                itmdt["data"].iloc[:, 0].max(),
                            ]
                        )
                        line_y = coeff_a * line_x + coeff_b
                        df_coeffs[f"{v_1}{v_2}x"] = line_x
                        df_coeffs[f"{v_1}{v_2}y"] = line_y
                        pbar.set_description(
                            desc=f"Computing Interactions: {v_1}-{v_2}"
                        )
                        pbar.update(1)
                df_coeffs.loc[:, "__x__"] = df_coeffs.iloc[:, 0]
                df_coeffs.loc[:, "__y__"] = df_coeffs.iloc[:, 1]
                itmdt = Intermediate(
                    coeffs=df_coeffs,
                    data=df_scatter,
                    visual_type="correlation_crossfilter",
                )
                rendered = render_correlation(itmdt)
                comps["interactions"] = components(rendered)
                pbar.set_description(desc="Formating Interactions")
            except Exception as error:
                comps["interactions"] = (0, {"error": error})
                pbar.set_description(desc="Something Happened...")
            pbar.update(1)
        else:
            comps["has_interaction"] = False
            pbar.set_description(desc="Skipping Interactions")
            pbar.update(len(num_cols) ** 2 + 1)

    return comps


# def format_full(
#     df: pd.DataFrame, comps: Dict[str, List[Tuple[str, str]]]
# ) -> Dict[str, Any]:
#     pass


# def format_mini(
#     df: pd.DataFrame, comps: Dict[str, List[Tuple[str, str]]]
# ) -> Dict[str, Any]:
#     pass


def _morph_figure(fig: Figure, **kwargs: Any) -> Figure:
    for key, value in kwargs.items():
        setattr(fig, key, value)
    return fig
