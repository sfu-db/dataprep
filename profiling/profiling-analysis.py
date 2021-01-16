##
import pandas as pd
import altair as alt
from json import loads

G = 1024 * 1024 * 1024
##
df = pd.read_json("results/report.1.json", lines=True)
df["DVM"] = df["mem_size"] / df["memory"]
df["MachineMem"] = df["memory"].apply(lambda value: f"{value / G:.1f}G")
df["DatasetMemSize"] = df["mem_size"].apply(lambda value: f"{value / G:.1f}G")
##
chartDVM = (
    alt.Chart(df[df["DVM"] < 10], title="plot_correlation(df) Comparison")
    .mark_line(point=True)
    .encode(
        y=alt.Y("elapsed", title="Elapsed (s)"),
        x=alt.X("DVM", title="Dataset Size / Memory Size"),
        color="name:N",
        tooltip=[
            alt.Tooltip("name:N"),
            alt.Tooltip("elapsed:Q", format=".0s"),
            alt.Tooltip("MachineMem"),
            alt.Tooltip("DatasetMemSize"),
        ],
        column=alt.Column("format", title="Data Format"),
        row=alt.Column("reader", title="Data Reader"),
    )
)
chartDVM
##
# chartBar = (
#     alt.Chart(
#         df[df.memory == 1 * G], title="Plot Comparison: 8G Mem/8 CPU/16 Data Partition"
#     )
#     .mark_bar()
#     .encode(
#         y="name:N",
#         x=alt.X("elapsed", title="Elapsed (s)"),
#         color="name",
#         tooltip="elapsed",
#         row="nrow:Q",
#         column=alt.Column("format:O", title="Data Format"),
#     )
#     .resolve_scale(x="independent")
# )
# chartBar
##
# pdf = df.pivot_table(
#     index=["Mem", "CPU", "Dataset", "Partition", "Row", "Col", "Mode"],
#     columns="Func",
#     values="Elapsed",
# ).reset_index()


##
