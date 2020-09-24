"""This module implements the generation of connector config generation UI."""


import json
from typing import Any, Dict, Generator, Optional, Tuple

from IPython.display import display
from ipywidgets import (
    HTML,
    Box,
    Button,
    Dropdown,
    GridspecLayout,
    HBox,
    Layout,
    Output,
    RadioButtons,
    Tab,
    Text,
    Textarea,
    VBox,
)

from .generator import ConfigGenerator


class ConfigGeneratorUI:  # pylint: disable=too-many-instance-attributes
    """Config Generator UI.

    Parameters
    ----------
    existing: Optional[Dict[str, Any]] = None
        Optionally pass in an existing configuration.
    save_file_name: Optional[str] = "tmp.json"
        The file name to save the config file.
    """

    grid: GridspecLayout
    box_layout: Layout
    textbox_layout: Layout
    item_layout: Layout
    cg_backend: ConfigGenerator

    # UI Boxes
    request_type: Dropdown
    url_area: Text
    params_box: Textarea
    authtype_box: RadioButtons
    authparams_box: Textarea
    pagtype_box: RadioButtons
    pagparams_box: Textarea
    dict_res: Dict[Any, Any]

    def __init__(
        self,
        existing: Optional[Dict[str, Any]] = None,
        save_file_name: Optional[str] = "tmp.json",
    ) -> None:
        self.box_layout = Layout(
            overflow="scroll hidden",
            border="1px solid black",
            width="600px",
            height="",
            flex_flow="row",
            display="flex",
        )
        self.textbox_layout = Layout(
            overflow="scroll hidden",
            border="1px solid black",
            width="10px",
            height="",
            flex_flow="row",
            display="flex",
        )
        self.tab_layout = Layout(
            #             border="1px solid black",
            width="600px"
        )
        self.label_layout = Layout(width="600px")
        self.textarea_layout = Layout(width="592px")
        self.smalltextarea_layout = Layout(width="570px")
        self.item_layout = Layout(height="200px", min_width="40px")
        self.cg_backend = ConfigGenerator(existing)
        self.save_file_name = save_file_name
        self.config = Textarea()
        self.tab = Tab()
        self.dict_res = {}
        self._make_grid()

    #     def make_title(self) -> HBox:
    #         """make UI title"""
    #         title = HTML(
    #             value=(
    #                 """<h1 style="font-family:Raleway, sans-serif; color:#444;margin:0px; """
    #                 """padding:10px"><b>Configuration Generator</b></h1>"""
    #             ),
    #             layout=Layout(width="600px"),
    #         )

    #         title_box = HBox([title])
    #         return title_box

    def _make_url(self) -> VBox:
        self.url_area = Text(
            disabled=False, placeholder="input requested url", layout={"width": "521px"}
        )
        self.request_type = Dropdown(
            options=["GET", "POST", "PUT"],
            value="GET",
            disabled=False,
            layout={"width": "max-content"},
        )

        items = [self.url_area, self.request_type]
        carousel_1 = Box(children=items, layout=self.box_layout)
        request_label = HTML(
            value=(
                """<h3 style="background-color:#E8E8E8; background-size: 100px; ">"""
                """<span style="color: #ff0000">1.</span> API URL</h3>"""
            ),
            layout=self.label_layout,
        )

        request_box = VBox([request_label, carousel_1])
        return request_box

    def _make_req_param(self) -> VBox:
        self.params_box = Textarea(
            placeholder=(
                "Please separate key and value by ':' ; while each key-value pair needs to be "
                "separated by ',' (e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout=self.textarea_layout,
        )

        params_label = HTML(
            value=(
                """<h3 style="background-color:#E8E8E8; background-size: 100px; ">
            <span style="color: #ff0000">2.</span> Request Parameters</h3>"""
            ),
            layout=self.label_layout,
        )

        carousel_2 = Box(children=[self.params_box], layout=self.box_layout)
        param_box = VBox([params_label, carousel_2])
        return param_box

    def _make_auth(self) -> VBox:
        self.authtype_box = RadioButtons(
            options=["No Authorization", "OAuth2", "QueryParam", "Bearer", "Header"],
            layout={"width": "max-content"},  # If the items' names are long
            description="",
            style={"description_width": "initial"},
            disabled=False,
        )

        auth_label = HTML(
            value=(
                """<h3 style="background-color:#E8E8E8; background-size: 100px; ">"""
                """<span style="color: #ff0000">3.</span> Authorization """
                """<span style="font-size: 14px">"""
                """<i>(some APIs require authorization)</i> </span></span></h3>"""
            ),
            layout=self.label_layout,
        )

        self.authparams_box = Textarea(
            placeholder=(
                "Please separate authtication key and corresponding value by ':' ; "
                "while each key-value pair needs to be separated by ',' "
                "(e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout=self.textarea_layout,
        )

        carousel_3 = Box(children=[self.authparams_box], layout=self.box_layout)
        auth_box = VBox([auth_label, self.authtype_box, carousel_3])
        return auth_box

    def _make_pag(self) -> VBox:
        self.pagtype_box = RadioButtons(
            options=["No Pagination", "offset", "seek", "page", "token"],
            layout={"width": "max-content"},  # If the items' names are long
            description="",
            style={"description_width": "initial"},
            disabled=False,
        )
        pag_label = HTML(
            value=(
                """<h3 style="background-color:#E8E8E8; background-size: 100px; ">
                <span style="color: #ff0000">4.</span> Pagination</h3>"""
            ),
            layout=self.label_layout,
        )

        self.pagparams_box = Textarea(
            placeholder=(
                "Please separate pagination key and corresponding value by ':' ;"
                " while each key-value pair needs to be separated by ',' "
                "(e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout=self.textarea_layout,
        )
        carousel_4 = Box(children=[self.pagparams_box], layout=self.box_layout)
        pag_box = VBox([pag_label, self.pagtype_box, carousel_4])
        return pag_box

    def _make_result(self) -> VBox:
        def on_button_clicked(remove: Any) -> None:
            if self.save_file_name is None:
                self.save_file_name = "tmp.json"
            file = open(self.save_file_name, "w")
            file.write(self.config.value)
            file.close()
            with saved_tooltip:
                print("Saved to %s callback removed %s!" % (self.save_file_name, remove))

        self.config = Textarea(
            value="",
            placeholder="",
            description="",
            disabled=False,
            layout=self.smalltextarea_layout,
        )

        save_button = Button(description="Save Config", layout=Layout(width="15%", height="35px"))
        save_button.style.button_color = "lightblue"
        save_button.style.font_weight = "740"
        save_button.on_click(on_button_clicked)

        saved_tooltip = Output(layout={"width": "initial"})
        config_box = self.config
        tab_children = [config_box]
        self.tab = Tab(layout=self.tab_layout, children=tab_children)
        description = ["Configuration File"]
        for i in range(len(tab_children)):
            self.tab.set_title(i, description[i])

        send_request_expl = HTML(
            value=(
                """<h4 style="font-family:Raleway, sans-serif; color:#444; """
                """font-weight:740">To send API URL request, please click on: </h4>"""
            ),
            layout=Layout(width="400px"),
        )
        send_request_button = Button(
            description="Send Request", layout=Layout(width="35%", height="35px")
        )
        send_request_button.style.button_color = "lightblue"
        send_request_button.style.font_weight = "740"
        send_request_button.on_click(self._on_send_request)
        request_and_review_label = HTML(
            value=(
                """<h3 style="background-color:#E8E8E8; background-size: 100px; ">"""
                """<span style="color: #ff0000">5.</span> Send Request & Review Results</h3>"""
            ),
            layout=self.label_layout,
        )
        result_box = VBox(
            [
                request_and_review_label,
                HBox(
                    [send_request_expl, send_request_button],
                    layout=Layout(width="520px"),
                ),
                self.tab,
                save_button,
            ]
        )
        return result_box

    def _make_grid(self) -> None:
        self.grid = GridspecLayout(40, 4, height="1200px", width="100Z%")
        self.grid[0:3, :] = self._make_url()
        self.grid[3:7, :] = self._make_req_param()
        self.grid[7:14, :] = self._make_auth()
        self.grid[14:21, :] = self._make_pag()
        self.grid[21:, :] = self._make_result()

    def _on_send_request(self, remove: Any) -> None:
        params_value = dict(_pairs(self.params_box.value))

        pagparams = None
        if self.pagtype_box.value != "No Pagination":
            pagparams = dict(_pairs(self.pagparams_box.value))
            pagparams["type"] = self.pagtype_box.value
            pagparams["maxCount"] = int(pagparams["maxCount"])

        authparams = dict(_pairs(self.authparams_box.value))
        sub_authparams, tokenparams = {}, {}
        for key, value in authparams.items():
            if key in ("client_id", "client_secret", "access_token"):
                tokenparams[key] = value
            else:
                sub_authparams[key] = value
        sub_authparams["type"] = self.authtype_box.value
        self.dict_res = {
            "url": self.url_area.value,
            "method": self.request_type.value,
            "params": params_value,
            "pagination": pagparams,
            "authorization": (sub_authparams, tokenparams)
            if self.authtype_box.value != "No Authentication"
            else None,
        }
        self.cg_backend.add_example(self.dict_res)
        if self.cg_backend.config.config is not None:
            self.config.value = json.dumps(
                self.cg_backend.config.config.dict(by_alias=True), indent=2
            )
        print("Send request, callback removed %s!" % remove)

    def display(self) -> None:
        """display UI"""
        display(self.grid)


def _pairs(content: str) -> Generator[Tuple[Any, Any], None, None]:
    """utility function to pair params"""
    if not content.strip():
        return

    for pair in content.split(","):

        x, *y = pair.split(":", maxsplit=1)
        if len(y) == 0:
            raise ValueError(f"Cannot parse pair {pair}")

        yield x.strip(), y[0].strip()
