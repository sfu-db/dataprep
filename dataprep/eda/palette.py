"""
This file defines palettes used for EDA.
"""
from bokeh.palettes import Category10, Category20, Greys256, Pastel1, viridis

BRG = ["#1f78b4", "#d62728", "#2ca02c"]
CATEGORY10 = Category10[10]
CATEGORY20 = Category20[20]
GREYS256 = Greys256
PASTEL1 = Pastel1[9]
VIRIDIS = viridis(256)
RDBU = [
    "#053061",
    "#063263",
    "#073466",
    "#083669",
    "#09386c",
    "#0a3a6f",
    "#0b3c72",
    "#0c3e75",
    "#0d4078",
    "#0e437b",
    "#0f457e",
    "#114781",
    "#124984",
    "#134b87",
    "#144d8a",
    "#154f8d",
    "#165190",
    "#175493",
    "#185695",
    "#195898",
    "#1a5a9b",
    "#1c5c9e",
    "#1d5ea1",
    "#1e60a4",
    "#1f62a7",
    "#2064aa",
    "#2166ac",
    "#2368ad",
    "#246aae",
    "#256caf",
    "#276db0",
    "#286fb0",
    "#2971b1",
    "#2b73b2",
    "#2c75b3",
    "#2d76b4",
    "#2f78b5",
    "#307ab6",
    "#317cb7",
    "#337db8",
    "#347fb9",
    "#3581b9",
    "#3783ba",
    "#3884bb",
    "#3986bc",
    "#3b88bd",
    "#3c8abe",
    "#3d8bbf",
    "#3f8dc0",
    "#408fc1",
    "#4191c2",
    "#4393c3",
    "#4694c4",
    "#4996c5",
    "#4c98c6",
    "#4f9ac7",
    "#529cc8",
    "#559ec9",
    "#58a0ca",
    "#5ba2cb",
    "#5ea4cc",
    "#61a6cd",
    "#65a8ce",
    "#68aacf",
    "#6bacd0",
    "#6eaed1",
    "#71b0d2",
    "#74b2d3",
    "#77b4d5",
    "#7ab6d6",
    "#7db8d7",
    "#80bad8",
    "#84bcd9",
    "#87beda",
    "#8ac0db",
    "#8dc2dc",
    "#90c4dd",
    "#93c5de",
    "#95c6df",
    "#98c8df",
    "#9ac9e0",
    "#9dcae1",
    "#9fcbe1",
    "#a2cde2",
    "#a4cee3",
    "#a7cfe4",
    "#a9d0e4",
    "#abd2e5",
    "#aed3e6",
    "#b0d4e6",
    "#b3d5e7",
    "#b5d7e8",
    "#b8d8e8",
    "#bad9e9",
    "#bddaea",
    "#bfdceb",
    "#c2ddeb",
    "#c4deec",
    "#c7dfed",
    "#c9e1ed",
    "#cce2ee",
    "#cee3ef",
    "#d1e5f0",
    "#d2e5f0",
    "#d3e6f0",
    "#d5e7f0",
    "#d6e7f1",
    "#d8e8f1",
    "#d9e9f1",
    "#dbe9f1",
    "#dceaf2",
    "#deebf2",
    "#dfecf2",
    "#e1ecf3",
    "#e2edf3",
    "#e4eef3",
    "#e5eef3",
    "#e7eff4",
    "#e8f0f4",
    "#eaf1f4",
    "#ebf1f4",
    "#edf2f5",
    "#eef3f5",
    "#f0f3f5",
    "#f1f4f6",
    "#f3f5f6",
    "#f4f5f6",
    "#f6f6f6",
    "#f7f6f6",
    "#f7f5f4",
    "#f7f4f2",
    "#f7f3f0",
    "#f8f2ee",
    "#f8f0ec",
    "#f8efea",
    "#f8eee8",
    "#f9ede7",
    "#f9ece5",
    "#f9ebe3",
    "#f9eae1",
    "#f9e9df",
    "#fae8dd",
    "#fae7db",
    "#fae5d9",
    "#fae4d7",
    "#fbe3d6",
    "#fbe2d4",
    "#fbe1d2",
    "#fbe0d0",
    "#fcdfce",
    "#fcdecc",
    "#fcddca",
    "#fcdcc8",
    "#fddbc7",
    "#fcd8c4",
    "#fcd6c1",
    "#fbd4be",
    "#fbd2bc",
    "#fbd0b9",
    "#faceb6",
    "#faccb4",
    "#facab1",
    "#f9c7ae",
    "#f9c5ab",
    "#f9c3a9",
    "#f8c1a6",
    "#f8bfa3",
    "#f8bda1",
    "#f7bb9e",
    "#f7b99b",
    "#f7b698",
    "#f6b496",
    "#f6b293",
    "#f5b090",
    "#f5ae8e",
    "#f5ac8b",
    "#f4aa88",
    "#f4a886",
    "#f4a683",
    "#f3a380",
    "#f2a07e",
    "#f19e7c",
    "#ef9b7a",
    "#ee9878",
    "#ed9676",
    "#ec9374",
    "#eb9072",
    "#ea8d70",
    "#e88b6e",
    "#e7886c",
    "#e6856a",
    "#e58368",
    "#e48065",
    "#e27d63",
    "#e17b61",
    "#e0785f",
    "#df755d",
    "#de725b",
    "#dd7059",
    "#db6d57",
    "#da6a55",
    "#d96853",
    "#d86551",
    "#d7624f",
    "#d6604d",
    "#d45d4b",
    "#d35a4a",
    "#d15749",
    "#d05447",
    "#ce5146",
    "#cd4f44",
    "#cc4c43",
    "#ca4942",
    "#c94641",
    "#c7433f",
    "#c6403e",
    "#c53e3c",
    "#c33b3b",
    "#c2383a",
    "#c03538",
    "#bf3237",
    "#be3036",
    "#bc2d34",
    "#bb2a33",
    "#b92732",
    "#b82431",
    "#b6212f",
    "#b51f2e",
    "#b41c2d",
    "#b2192b",
    "#b0172a",
    "#ad162a",
    "#aa1529",
    "#a71429",
    "#a41328",
    "#a11228",
    "#9e1127",
    "#9b1027",
    "#991027",
    "#960f26",
    "#930e26",
    "#900d25",
    "#8d0c25",
    "#8a0b24",
    "#870a24",
    "#840923",
    "#810823",
    "#7e0722",
    "#7b0622",
    "#780521",
    "#750421",
    "#720320",
    "#6f0220",
    "#6c011f",
    "#69001f",
    "#67001f",
]
