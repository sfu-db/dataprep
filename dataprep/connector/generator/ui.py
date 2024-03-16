"""This module implements the generation of connector config generation UI."""

from base64 import b64encode
from typing import Any, Dict, Generator, Optional, Tuple
from zipfile import ZipFile
from io import BytesIO

from IPython.display import display, Javascript
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
    Label,
    Text,
    Textarea,
    VBox,
)

from .generator import ConfigGenerator

BOX_LAYOUT = Layout(
    overflow="scroll hidden",
    border="1px solid black",
    width="100%",
    flex_flow="row",
    display="flex",
)


class ConfigGeneratorUI:  # pylint: disable=too-many-instance-attributes
    """Config Generator UI.

    Parameters
    ----------
    existing: Optional[Dict[str, Any]] = None
        Optionally pass in an existing configuration.
    """

    grid: GridspecLayout
    existing: Optional[Dict[str, Any]] = None

    # UI Boxes
    request_type: Dropdown
    url_area: Text
    params_box: Textarea
    authtype_box: RadioButtons
    authparams_box: Textarea
    pagtype_box: RadioButtons
    pagparams_box: Textarea
    table_path_box: Text
    output: Output

    def __init__(
        self,
        existing: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.existing = existing
        self.output = Output()

        self._make_grid()

    def _make_url(self) -> VBox:
        self.url_area = Text(disabled=False, placeholder="URL", layout={"width": "100%"})
        self.request_type = Dropdown(
            options=["GET", "POST", "PUT"],
            value="GET",
            disabled=False,
            layout={"width": "max-content"},
        )

        request_box = VBox(
            [
                _make_header(1, "API URL"),
                Box(children=[self.url_area, self.request_type], layout=BOX_LAYOUT),
            ]
        )
        return request_box

    def _make_req_param(self) -> VBox:
        self.params_box = Textarea(
            placeholder=(
                "Please separate key and value by ':' ; while each key-value pair needs to be "
                "separated by ',' (e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout={"width": "100%", "height": "100%"},
        )

        carousel_2 = Box(children=[self.params_box], layout=BOX_LAYOUT)
        param_box = VBox([_make_header(2, "Request Parameters"), carousel_2])
        return param_box

    def _make_auth(self) -> VBox:
        self.authtype_box = RadioButtons(
            options=["No Authorization", "OAuth2", "QueryParam", "Bearer", "Header"],
            layout={"width": "max-content"},  # If the items' names are long
            description="",
            style={"description_width": "initial"},
            disabled=False,
        )

        self.authparams_box = Textarea(
            placeholder=(
                "Please separate authtication key and corresponding value by ':' ; "
                "while each key-value pair needs to be separated by ',' "
                "(e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout={"width": "100%"},
        )

        carousel_3 = Box(children=[self.authparams_box], layout=BOX_LAYOUT)
        auth_box = VBox([_make_header(3, "Authorization"), self.authtype_box, carousel_3])
        return auth_box

    def _make_pag(self) -> VBox:
        self.pagtype_box = RadioButtons(
            options=["No Pagination", "offset", "seek", "page", "token"],
            layout={"width": "max-content"},  # If the items' names are long
            description="",
            style={"description_width": "initial"},
            disabled=False,
        )

        self.pagparams_box = Textarea(
            placeholder=(
                "Please separate pagination key and corresponding value by ':' ;"
                " while each key-value pair needs to be separated by ',' "
                "(e.g. name:abcdefg, date:2019-12-12)"
            ),
            layout={"width": "100%"},
        )
        carousel_4 = Box(children=[self.pagparams_box], layout=BOX_LAYOUT)
        pag_box = VBox([_make_header(4, "Pagination"), self.pagtype_box, carousel_4])
        return pag_box

    def _make_result(self) -> VBox:
        send_button = Button(
            description="Send Request & Download",
            layout=Layout(width="35%", height="35px"),
            style={"button_color": "lightblue", "font_weight": "740"},
        )
        send_button.on_click(self._on_send_request)

        return VBox(
            [_make_header(6, "Send Request & Download"), send_button, self.output],
            layout={"align_items": "center"},
        )

    def _make_generator_option(self) -> VBox:

        self.table_path_box = Text(
            disabled=False, placeholder="table_path", layout={"width": "90%"}
        )

        return VBox(
            [
                _make_header(5, "Generator Options"),
                HBox([Label(value="Table Path:"), self.table_path_box], layout=BOX_LAYOUT),
            ],
            layout={"align_items": "center"},
        )

    def _make_grid(self) -> None:
        self.grid = GridspecLayout(
            32,
            4,
            height="960px",
            width="800px",
            layout={"border": "1px solid black", "padding": "5px"},
        )
        self.grid[0:4, :] = self._make_url()
        self.grid[4:8, :] = self._make_req_param()
        self.grid[8:16, :] = self._make_auth()
        self.grid[16:24, :] = self._make_pag()
        self.grid[24:28, :] = self._make_generator_option()
        self.grid[28:32, :] = self._make_result()

    def _on_send_request(self, _: Any) -> None:
        params_value = dict(_pairs(self.params_box.value))

        pagparams = None
        if self.pagtype_box.value != "No Pagination":
            pagparams = dict(_pairs(self.pagparams_box.value))
            pagparams["type"] = self.pagtype_box.value
            pagparams["maxCount"] = int(pagparams["maxCount"])

        authparams = None
        if self.authtype_box.value != "No Authorization":
            authparams = dict(_pairs(self.authparams_box.value))
            authparams_user = {}
            for key, value in authparams.items():
                if key in ("client_id", "client_secret", "access_token"):
                    authparams_user[key] = value
            for key in ("client_id", "client_secret", "access_token"):
                del authparams[key]
            authparams["type"] = self.authtype_box.value

        example = {
            "url": self.url_area.value,
            "method": self.request_type.value,
            "params": params_value,
            "pagination": pagparams,
            "authorization": (
                (authparams, authparams_user)
                if self.authtype_box.value != "No Authorization"
                else None
            ),
        }

        backend = ConfigGenerator(self.existing)
        with self.output:
            if self.table_path_box.value == "":
                backend.add_example(example)
            else:
                backend.add_example(example, self.table_path_box.value)

            # Download the config
            if backend.config.config is None:
                raise RuntimeError("Not possible")

            payload = backend.config.config.json(by_alias=True)
            data = BytesIO()

            with ZipFile(data, mode="w") as zipf:
                zipf.writestr("dblp/_meta.json", """"{"tables": ["publication"]}""")
                zipf.writestr("dblp/publication.json", payload)

            payload = b64encode(data.getvalue()).decode()
            script = (
                "let newlink = document.createElement('a');"
                f"newlink.setAttribute('href', 'data:application/zip;base64,{payload}');"
                f"newlink.setAttribute('target', '_blank');"
                f"newlink.setAttribute('download', 'dblp.zip');"
                "newlink.click();"
            )
            display(Javascript(script))

    def display(self) -> None:
        """display UI"""
        display(self.grid)


def _make_header(seq: int, title: str) -> HTML:
    """make a header"""

    return HTML(
        value=(
            """<h3 style="background-color:#E8E8E8; background-size: 100px; ">"""
            f"""<span style="color: #ff0000">{seq}.</span>{title}</h3>"""
        ),
        layout=Layout(width="100%"),
    )


def _pairs(content: str) -> Generator[Tuple[Any, Any], None, None]:
    """utility function to pair params"""
    if not content.strip():
        return

    for pair in content.split(","):

        x, *y = pair.split(":", maxsplit=1)
        if len(y) == 0:
            raise ValueError(f"Cannot parse pair {pair}")

        yield x.strip(), y[0].strip()
