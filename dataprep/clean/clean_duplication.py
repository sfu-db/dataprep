"""
Clean a DataFrame column containing duplicate values.
"""

from typing import Tuple, List, Dict, Any, Union

from ipywidgets.widgets import Label, Dropdown, Checkbox, Button, HBox, VBox, Box, Layout, Text
import pandas as pd
import dask.dataframe as dd
from varname import argname

from .clean_duplication_utils import Clusterer

DEFAULT_NGRAM = "2"
DEFAULT_RADIUS = "2"
DEFAULT_BLOCK_SIZE = "6"


def clean_duplication(
    df: Union[pd.DataFrame, dd.DataFrame],
    column: str,
    df_var_name: str = "default",
    page_size: int = 5,
) -> Box:
    """
    Cleans and standardizes duplicate values in a DataFrame.

    Read more in the :ref:`User Guide <duplication_userguide>`.

    Parameters
    ----------
    df
        A pandas or Dask DataFrame containing the data to be cleaned.
    column
        The name of the column containing duplicate values.
    df_var_name
        Optional parameter containing the variable name of the DataFrame being cleaned.
        This is only needed for legacy compatibility with the original veraion of this
        function, which needed it to produce correct exported code.

        (default: 'default')
    page_size
        The number of clusters to display on each page.

        (default: 5)

    Examples
    --------

    After running clean_duplication(df, 'city') below in a notebook, a GUI will appear.
    Select the merge checkbox, press merge and re-cluster, then press finish.

    >>> df = pd.DataFrame({'city': ['New York', 'new york']})
    >>> clean_duplication(df, 'city')

          city
    0    New York
    1    New York
    """
    if df_var_name == "default":
        df_var_name = argname("df", func=clean_duplication)

    return UserInterface(df, column, df_var_name, page_size).display()


class UserInterface:
    """
    A user interface used by the clean_duplication function.
    """

    # pylint: disable=too-many-instance-attributes

    _clusterer: Clusterer
    _page_size: int
    _clustering_method_label: Label
    _clustering_method_drop: Dropdown
    _export_code: Checkbox
    _sel_all: Checkbox
    _next_button: Button
    _prev_button: Button
    _page_pos: int
    _ngram_text: Text
    _radius_text: Text
    _block_chars_text: Text
    _dropds: HBox
    _reprs: List[Text]
    _checks: List[Checkbox]
    _cluster_vbox: VBox
    _box: Box
    _loading_label: Label
    _invalid_param_label: Label

    def __init__(self, df: pd.DataFrame, col_name: str, df_name: str, page_size: int):
        self._clusterer = Clusterer(df, col_name, df_name)
        self._clusterer.cluster("fingerprint")

        self._page_size = page_size

        # clustering dropdown and export code checkbox, used in the top row
        self._clustering_method_label = Label(
            " Clustering Method: ", layout=Layout(margin="2px 0 0 20px")
        )
        self._clustering_method_drop = Dropdown(
            options=["fingerprint", "ngram-fingerprint", "phonetic-fingerprint", "levenshtein"],
            layout=Layout(width="150px", margin="0 0 0 10px"),
        )
        self._clustering_method_drop.observe(self._cluster_method_change, names="value")
        self._export_code = Checkbox(
            value=True,
            description="export code",
            layout=Layout(width="165px", margin="0 0 0 482px"),
            style={"description_width": "initial"},
        )
        self._dropds = HBox(
            [
                self._clustering_method_label,
                self._clustering_method_drop,
                self._export_code,
            ],
            layout=Layout(height="35px", margin="10px 0 0 0"),
        )
        # text boxes for clustering parameters used in the top row
        self._ngram_text = Text(
            value=DEFAULT_NGRAM,
            description="n-gram",
            layout=Layout(width="130px"),
            continuous_update=False,
        )
        self._radius_text = Text(
            value=DEFAULT_RADIUS,
            description="Radius",
            layout=Layout(width="130px"),
            continuous_update=False,
        )
        self._block_chars_text = Text(
            value=DEFAULT_BLOCK_SIZE,
            description="Block Chars",
            layout=Layout(width="130px"),
            continuous_update=False,
        )
        self._ngram_text.observe(self._param_recluster, names="value")
        self._radius_text.observe(self._param_recluster, names="value")
        self._block_chars_text.observe(self._param_recluster, names="value")

        # create header labels, second row
        headers = HBox(
            [
                Label("Distinct values", layout=Layout(margin="0 0 0 10px")),
                Label("Total values", layout=Layout(margin="0 0 0 35px")),
                Label("Cluster values", layout=Layout(margin="0 0 0 95px")),
                Label("Merge?", layout=Layout(margin="0 0 0 295px")),
                Label("Representative value", layout=Layout(margin="0 0 0 50px")),
            ],
            layout=Layout(margin="10px"),
        )

        # create buttons for bottom row
        self._sel_all = Checkbox(description="Select all", layout=Layout(width="165px"))
        self._sel_all.observe(self._select_all, names="value")

        merge_and_recluster = Button(
            description="Merge and Re-Cluster", layout=Layout(margin="0 0 0 466px", width="150px")
        )
        merge_and_recluster.on_click(self._execute_merge)

        finish = Button(description="Finish", layout=Layout(margin="0 0 0 10px"))
        finish.on_click(self._close)

        # next and previous page buttons
        self._next_button = Button(description="Next")
        self._next_button.on_click(self._next_page)

        self._prev_button = Button(description="Previous", layout=Layout(margin="0 0 0 20px"))
        self._prev_button.on_click(self._prev_page)

        # an index in the clusters Series indicating the start of the current page
        self._page_pos = 0
        # loading label, displayed when re-clustering or next page load
        self._loading_label = Label("Loading...", layout=Layout(margin="170px 0 0 440px"))
        # displayed when the user enters a non integer value into a clustering parameter text box
        self._invalid_param_label = Label(
            "Invalid clustering parameter, please enter an integer",
            layout=Layout(margin="170px 0 0 350px"),
        )

        self._reprs = [
            Text(layout=Layout(width="200px", margin="0 10px 0 40px"))
            for _ in range(self._page_size)
        ]
        self._checks = [
            Checkbox(indent=False, layout=Layout(width="auto", margin="0 0 0 20px"))
            for _ in range(self._page_size)
        ]

        # VBox containing a VBox with all the clusters in the first row and an optional
        # second row containing next and previous page buttons
        self._cluster_and_next_prev = VBox()
        self._cluster_vbox = VBox(layout=Layout(height="450px", flex_flow="row wrap"))

        footer = HBox([self._sel_all, merge_and_recluster, finish])

        box_children = [self._dropds, headers, self._cluster_and_next_prev, footer]

        box_layout = Layout(
            display="flex", flex_flow="column", align_items="stretch", border="solid"
        )
        self._box = Box(children=box_children, layout=box_layout)
        self._update_clusters()

    def _update_clusters(self) -> None:
        """
        Updates the clusters currently being displayed.
        """
        line = HBox(children=[Label("-" * 186, layout=Layout(margin="0 0 0 18px"))])
        self._sel_all.value = False

        cluster_page = self._clusterer.get_page(self._page_pos, self._page_pos + self._page_size)

        label_layout = Layout(height="22px", width="360px")
        box_children = [line]
        for idx, cluster in enumerate(cluster_page):
            labels = []
            for cluster_val, cnt in cluster:
                if cnt > 1:
                    cluster_val += f" ({cnt} rows)"
                labels.append(Label(cluster_val, layout=label_layout))

            totals_vals = sum(cnt for _, cnt in cluster)
            distinct_vals = len(cluster)

            self._reprs[idx].value = cluster[0][0]
            self._checks[idx].value = False
            box_children.append(
                HBox(
                    [
                        Label(str(distinct_vals), layout=Layout(width="60px", margin="0 0 0 60px")),
                        Label(str(totals_vals), layout=Layout(width="60px", margin="0 0 0 50px")),
                        VBox(children=labels, layout=Layout(margin="0 0 0 80px")),
                        self._checks[idx],
                        self._reprs[idx],
                    ]
                )
            )

            box_children.append(line)

        # no clusters to display
        if len(cluster_page) == 0:
            box_children = [
                Label(
                    "No clusters, try a different clustering method",
                    layout=Layout(margin="170px 0 0 360px"),
                )
            ]

        self._cluster_vbox.children = box_children
        cluster_and_next_prev = [self._cluster_vbox]
        self._add_next_prev_button_row(cluster_and_next_prev)
        self._cluster_and_next_prev.children = cluster_and_next_prev

    def _update_dropds(self, clustering_method: str) -> None:
        """
        Update the dropdowns row of the UI to display the required text boxes for passing
        parameters needed for the given clustering method.
        """
        if clustering_method in ("fingerprint", "phonetic-fingerprint"):
            self._export_code.layout.margin = "0 0 0 482px"
            self._dropds.children = [
                self._clustering_method_label,
                self._clustering_method_drop,
                self._export_code,
            ]

        if clustering_method == "ngram-fingerprint":
            self._export_code.layout.margin = "0 0 0 348px"
            self._dropds.children = [
                self._clustering_method_label,
                self._clustering_method_drop,
                self._ngram_text,
                self._export_code,
            ]

        if clustering_method == "levenshtein":
            self._export_code.layout.margin = "0 0 0 214px"
            self._dropds.children = [
                self._clustering_method_label,
                self._clustering_method_drop,
                self._radius_text,
                self._block_chars_text,
                self._export_code,
            ]

    def _param_recluster(self, _: Dict[str, Any]) -> None:
        """
        Re-cluster the dataframe with the new clustering parameters.
        Triggered when the value in a clustering parameter textbox is changed.
        """
        self._display_message(self._loading_label)
        try:
            self._clusterer.set_cluster_params(*self._cluster_params())
            cluster_method = self._clustering_method_drop.value
            self._clusterer.cluster(cluster_method)
            self._page_pos = 0
            self._update_clusters()
        except ValueError:
            self._display_message(self._invalid_param_label)

    def _cluster_params(self) -> Tuple[int, int, int]:
        """
        Retrieve clustering parameters from their respective text boxes.
        """
        ngram = self._ngram_text.value if self._ngram_text.value else DEFAULT_NGRAM
        radius = self._radius_text.value if self._radius_text.value else DEFAULT_RADIUS
        block_size = self._block_chars_text.value if self._block_chars_text else DEFAULT_BLOCK_SIZE
        return int(ngram), int(radius), int(block_size)

    def _select_all(self, change: Dict[str, Any]) -> None:
        """
        Triggered when the select all checkbox is selected or unselected.
        Changes the value of the cluster checkboxes to match the state of the select all checkbox.
        """
        for check in self._checks:
            check.value = change["new"]

    def _cluster_method_change(self, change: Dict[str, Any]) -> None:
        """
        Triggered when the cluster method dropdown state is changed.
        Re-clusters the DataFrame with the new clustering method.
        """
        self._update_dropds(change["new"])
        self._display_message(self._loading_label)
        cluster_method = self._clustering_method_drop.value
        self._clusterer.cluster(cluster_method)
        self._page_pos = 0
        self._update_clusters()

    def _add_next_prev_button_row(self, box_children: List[Union[HBox, VBox]]) -> None:
        """
        Adds a next page or previous page button, if the operation is valid.
        """
        next_prev = []
        prev_is_valid = self._page_pos - self._page_size >= 0
        next_is_valid = self._page_pos + self._page_size < len(self._clusterer.clusters)

        if prev_is_valid and next_is_valid:
            self._next_button.layout.margin = "0 0 0 628px"
            next_prev.append(self._prev_button)
            next_prev.append(self._next_button)

        elif prev_is_valid:
            next_prev.append(self._prev_button)

        elif next_is_valid:
            self._next_button.layout.margin = "0 0 0 795px"
            next_prev.append(self._next_button)

        if next_is_valid or prev_is_valid:
            box_children.append(HBox(next_prev, layout={"height": "50px"}))

    def _next_page(self, _: Dict[str, Any]) -> None:
        """
        Display the next page of clusters by increasing the page position.
        """
        self._display_message(self._loading_label)
        self._page_pos += self._page_size
        self._update_clusters()
        self._sel_all_on_page()

    def _prev_page(self, _: Dict[str, Any]) -> None:
        """
        Display the previous page of clusters by decreasing the page position.
        """
        self._display_message(self._loading_label)
        self._page_pos -= self._page_size
        self._update_clusters()
        self._sel_all_on_page()

    def _display_message(self, message_label: Label) -> None:
        """
        Display a message to the user, used for the loading screen
        and invalid clustering parameter screen
        """
        self._cluster_vbox.children = [message_label]
        # don't display next and prev buttons
        self._cluster_and_next_prev.children = [self._cluster_and_next_prev.children[0]]

    def _sel_all_on_page(self) -> None:
        """
        Select all checkboxes on the current page of clusters if the
        select all checkbox is selected.
        """
        if self._sel_all.value:
            for check in self._checks:
                check.value = True

    def _close(self, _: Dict[str, Any]) -> None:
        """
        Close the UI and display the final dataframe in the next jupyter notebook cell.
        """
        self._clusterer.final_df()
        self._box.close()

    def _execute_merge(self, _: Dict[str, Any]) -> None:
        """
        Merge the selected clusters and re-cluster the dataframe. If the export code checkbox is
        selected the required replace function calls and add them to the jupyter notebook cell.
        """
        self._display_message(self._loading_label)

        do_merge = [check.value for check in self._checks]
        new_values = [text.value for text in self._reprs]
        cluster_page = self._clusterer.get_page(self._page_pos, self._page_pos + self._page_size)

        if self._export_code.value:
            self._clusterer.live_export_code(cluster_page, do_merge, new_values)

        cluster_method = self._clustering_method_drop.value
        self._clusterer.execute_merge_code(cluster_page, do_merge, new_values)
        self._clusterer.cluster(cluster_method)
        self._update_page_if_empty()
        self._update_clusters()

    def _update_page_if_empty(self) -> None:
        """
        Decrease the page if the last page is empty.
        Needed for when all clusters on the last page are merged.
        """
        if self._page_pos >= len(self._clusterer.clusters):
            self._page_pos -= self._page_size

    def display(self) -> Box:
        """Display the UI."""
        return self._box
