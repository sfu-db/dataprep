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

from ..basic import compute, render
from ..basic.compute import calc_stats
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

    num_col: List[str] = []
    for col in df.columns:
        if is_dtype(detect_dtype(df[col]), Continuous()):
            num_col.append(col)

    with tqdm(total=7 + 2 * len(df.columns) + len(num_col) ** 2) as pbar:
        # Missing Values
        itmdt = compute_missing(df)
        pbar.set_description(desc="Computing Missing Values")
        pbar.update(1)
        if any(itmdt["data_total_missing"].values()):
            try:
                comps["has_missing"] = True
                rendered = render_missing(itmdt)
                comps["missing"] = components(
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
                pbar.set_description(desc="Formating Missing Values")
                pbar.update(1)
            except Exception as error:
                comps["has_missing"] = True
                comps["missing"] = (0, {"error": error})  # same template for rendering
                pbar.set_description(desc="Something Happened...")
                pbar.update(1)
        else:
            comps["has_missing"] = False
            pbar.set_description(desc="Skipping Missing Values")
            pbar.update(1)

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

        stats = calc_stats(df, counter)
        pbar.set_description(desc="Computing Overview")
        pbar.update(1)
        comps["overview"] = stats
        pbar.set_description(desc="Formating Overview")
        pbar.update(1)

        # Variables
        comps["variables"] = {}
        for col in df.columns:
            try:
                itmdt = compute(df, col, top_words=15)
                pbar.set_description(desc=f"Computing {col}")
                pbar.update(1)
                rendered = render(itmdt)
                comps["variables"][col] = {
                    "tabledata": itmdt["statsdata"],
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
                pbar.set_description(desc=f"Something Happened...")
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
        if len(num_col) > 1:
            comps["has_interaction"] = True
            # set initial x,y axis value
            df_scatter = df.loc[:, num_col]
            df_scatter.loc[:, "__x__"] = df_scatter.iloc[:, 0]
            df_scatter.loc[:, "__y__"] = df_scatter.iloc[:, 0]
            try:
                for v_1 in num_col:
                    for v_2 in num_col:
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
                pbar.update(1)
            except Exception as error:
                comps["interactions"] = (0, {"error": error})
                pbar.set_description(desc="Something Happened...")
                pbar.update(1)
        else:
            comps["has_interaction"] = False
            pbar.set_description(desc="Skipping Interactions")
            pbar.update(len(num_col) ** 2 + 1)

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
