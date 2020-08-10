"""
    This module implements the formatting
    for create_report(df) function.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from bokeh.embed import components
from bokeh.models import Title
from bokeh.plotting import Figure

from ..distribution import compute, render
from ..distribution.compute import calc_stats
from ..distribution.render import _format_values
from ..correlation import compute_correlation, render_correlation
from ..dtypes import Continuous, DateTime, Nominal, detect_dtype, is_dtype
from ..intermediate import Intermediate
from ..missing import compute_missing, render_missing
from ..utils import is_notebook

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
        raise ValueError(f"Unknown mode: {mode}")
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
    num_col: List[str] = []
    for col in df.columns:
        if is_dtype(detect_dtype(df[col]), Continuous()):
            num_col.append(col)

    with tqdm(
        total=7 + 2 * len(df.columns) + len(num_col) ** 2, dynamic_ncols=True
    ) as pbar:
        # Missing Values
        pbar.set_description(desc="Computing Missing Values")
        pbar.update(1)
        itmdt = compute_missing(df)
        if any(itmdt["data_total_missing"].values()):
            try:
                pbar.set_description(desc="Formating Missing Values")
                pbar.update(1)
                comps["has_missing"] = True
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
            except Exception as error:
                pbar.set_description(desc="Something Happened...")
                pbar.update(1)
                comps["has_missing"] = True
                comps["missing"] = (0, {"error": error})  # same template for rendering
        else:
            pbar.set_description(desc="Skipping Missing Values")
            pbar.update(1)
            comps["has_missing"] = False

        # Overview
        counter = {"Categorical": 0, "Numerical": 0, "Datetime": 0}
        for column in df.columns:
            column_dtype = detect_dtype(df[column])
            if is_dtype(column_dtype, Nominal()):
                counter["Categorical"] += 1
            elif is_dtype(column_dtype, Continuous()):
                counter["Numerical"] += 1
            elif is_dtype(column_dtype, DateTime()):
                counter["Datetime"] += 1

        pbar.set_description(desc="Computing Overview")
        pbar.update(1)
        stats = calc_stats(df, counter)

        pbar.set_description(desc="Formating Overview")
        pbar.update(1)

        comps["overview"] = _format_stats(stats, "overview")

        # Variables
        comps["variables"] = {}
        for col in df.columns:
            try:
                pbar.set_description(desc=f"Computing {col}")
                pbar.update(1)
                itmdt = compute(df, col, top_words=15)

                pbar.set_description(desc=f"Formating {col}")
                pbar.update(1)
                rendered = render(itmdt)
                data = itmdt["stats"]
                if is_dtype(detect_dtype(df[col]), Continuous()):
                    stats = _format_stats(data, "var_num")
                elif is_dtype(detect_dtype(df[col]), Nominal()):
                    stats = _format_stats(data, "var_cat")
                elif is_dtype(detect_dtype(df[col]), DateTime()):
                    stats = _format_stats(data, "var_dt")
                else:
                    raise TypeError(f"Unsupported dtype: {detect_dtype(df[col])}")

                comps["variables"][col] = {
                    "tabledata": stats,
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
            except Exception as error:
                pbar.set_description(desc=f"Something Happened...")
                pbar.update(2)
                comps["variables"][col] = {"error": error, "plots": (0, 0)}

        # Correlations
        try:
            pbar.set_description(desc="Computing Correlations")
            pbar.update(1)
            itmdt = compute_correlation(df)
            if len(itmdt) != 0:
                pbar.set_description(desc="Formating Correlations")
                pbar.update(1)
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
            else:
                pbar.set_description(desc="Skipping Correlations")
                pbar.update(1)
                comps["has_correlation"] = False
        except Exception as error:
            pbar.set_description(desc="Something Happened...")
            pbar.update(2)
            comps["has_correlation"] = True
            comps["correlations"] = (0, {"error": error})  # same template for rendering

        # Interactions
        df_coeffs: pd.DataFrame = pd.DataFrame({})
        if len(num_col) > 1:
            comps["has_interaction"] = True
            # set initial x,y axis value
            df_scatter = df.loc[:, num_col]
            df_scatter.loc[:, "__x__"] = df_scatter.iloc[:, 0]
            df_scatter.loc[:, "__y__"] = df_scatter.iloc[:, 0]
            try:
                for v_1 in num_col:
                    for v_2 in num_col:
                        pbar.set_description(
                            desc=f"Computing Interactions: {v_1}-{v_2}"
                        )
                        pbar.update(1)
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
                df_coeffs.loc[:, "__x__"] = df_coeffs.iloc[:, 0]
                df_coeffs.loc[:, "__y__"] = df_coeffs.iloc[:, 1]
                itmdt = Intermediate(
                    coeffs=df_coeffs,
                    data=df_scatter,
                    visual_type="correlation_crossfilter",
                )
                pbar.set_description(desc="Formating Interactions")
                pbar.update(1)
                rendered = render_correlation(itmdt)
                comps["interactions"] = components(
                    _morph_figure(rendered, sizing_mode="stretch_width")
                )
            except Exception as error:
                pbar.set_description(desc="Something Happened...")
                pbar.update(1)
                comps["interactions"] = (0, {"error": error})
        else:
            pbar.set_description(desc="Skipping Interactions")
            pbar.update(len(num_col) ** 2 + 1)
            comps["has_interaction"] = False
        pbar.set_description(desc="Report has been created!")
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


def _format_stats(stats: Any, at: str,) -> Any:
    # pylint: disable=too-many-locals, invalid-name
    if at == "overview":
        (
            nrows,
            ncols,
            npresent_cells,
            nrows_wo_dups,
            mem_use,
            dtypes_cnt,
        ) = stats.values()
        ncells = nrows * ncols
        data = {
            "Number of Variables": ncols,
            "Number of Observations": nrows,
            "Missing Cells": float(ncells - npresent_cells),
            "Missing Cells (%)": 1 - (npresent_cells / ncells),
            "Duplicate Rows": nrows - nrows_wo_dups,
            "Duplicate Rows (%)": 1 - (nrows_wo_dups / nrows),
            "Total Size in Memory": float(mem_use),
            "Average Record Size in Memory": mem_use / nrows,
        }
        data = {k: _format_values(k, v) for k, v in data.items()}
        return data, dtypes_cnt
    elif at == "var_num":
        overview = {
            "Distinct Count": stats["nunique"],
            "Unique (%)": stats["nunique"] / stats["npresent"],
            "Missing": stats["nrows"] - stats["npresent"],
            "Missing (%)": 1 - (stats["npresent"] / stats["nrows"]),
            "Infinite": stats["ninfinite"],
            "Infinite (%)": stats["ninfinite"] / stats["nrows"],
            "Mean": stats["mean"],
            "Minimum": stats["min"],
            "Maximum": stats["max"],
            "Zeros": stats["nzero"],
            "Zeros (%)": stats["nzero"] / stats["nrows"],
            "Memory Size": stats["mem_use"],
        }
        quantile = {
            "Minimum": stats["min"],
            "5-th Percentile": stats["qntls"].iloc[5],
            "Q1": stats["qntls"].iloc[25],
            "Median": stats["qntls"].iloc[50],
            "Q3": stats["qntls"].iloc[75],
            "95-th Percentile": stats["qntls"].iloc[95],
            "Maximum": stats["max"],
            "Range": stats["max"] - stats["min"],
            "IQR": stats["qntls"].iloc[75] - stats["qntls"].iloc[25],
        }
        descriptive = {
            "Standard Deviation": stats["std"],
            "Coefficient of Variation": stats["std"] / stats["mean"]
            if stats["mean"] != 0
            else np.nan,
            "Kurtosis": float(stats["kurt"]),
            "Mean": stats["mean"],
            "Skewness": float(stats["skew"]),
            "Sum": stats["mean"] * stats["npresent"],
            "Variance": stats["std"] ** 2,
        }
        overview = {k: _format_values(k, v) for k, v in overview.items()}
        quantile = {k: _format_values(k, v) for k, v in quantile.items()}
        descriptive = {k: _format_values(k, v) for k, v in descriptive.items()}
        return overview, quantile, descriptive
    elif at == "var_cat":
        stats, length_stats, letter_stats = stats
        ov_stats = {
            "Distinct Count": stats["nunique"],
            "Unique (%)": stats["nunique"] / stats["npresent"],
            "Missing": stats["nrows"] - stats["npresent"],
            "Missing (%)": 1 - stats["npresent"] / stats["nrows"],
            "Memory Size": stats["mem_use"],
        }
        sampled_rows = (
            "1st row",
            "2nd row",
            "3rd row",
            "4th row",
            "5th row",
        )
        smpl = dict(zip(sampled_rows, stats["first_rows"]))

        ov_stats = {k: _format_values(k, v) for k, v in ov_stats.items()}
        length_stats = {k: _format_values(k, v) for k, v in length_stats.items()}
        smpl = {k: f"{v[:18]}..." if len(v) > 18 else v for k, v in smpl.items()}
        letter_stats = {k: _format_values(k, v) for k, v in letter_stats.items()}
        return ov_stats, length_stats, smpl, letter_stats
    elif at == "var_dt":
        return ({k: _format_values(k, v) for k, v in stats.items()},)
    else:
        raise ValueError("Unknown section")
