"""
Flask backend of Dataprep.Clean GUI.
"""
from typing import Any

import threading
import pandas as pd


from flask import Flask, render_template, request
from flask_cors import CORS

from dataprep.clean import (
    clean_email,
    clean_headers,
    clean_country,
    clean_date,
    clean_lat_long,
    clean_ip,
    clean_phone,
    clean_text,
    clean_url,
    clean_address,
    clean_df,
)


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__, static_folder="frontend_dist/static", template_folder="frontend_dist")
app.config.from_object(__name__)

# enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})


# pylint: disable=C0103, W0603
index_df = pd.DataFrame({})

# Dictionary of functions
clean_function_dic = {
    "clean_email": clean_email,
    "clean_headers": clean_headers,
    "clean_country": clean_country,
    "clean_date": clean_date,
    "clean_lat_long": clean_lat_long,
    "clean_ip": clean_ip,
    "clean_phone": clean_phone,
    "clean_text": clean_text,
    "clean_url": clean_url,
    "clean_address": clean_address,
    "clean_df": clean_df,
}
# sanity check route
@app.route("/", methods=["GET"])
def index() -> Any:
    """
    This function shows the index of GUI.
    """
    return render_template("index.html")


@app.route("/getInitSheet", methods=["GET"])
def getInitSheet() -> Any:
    """
    This function shows the input table in GUI.
    """
    global index_df
    index_df = index_df.astype(str)
    col_names = index_df.columns.values.tolist()
    table_columns = []
    for col_name in col_names:
        temp_dic = {}
        temp_dic["colName"] = col_name
        temp_dic["colLabel"] = col_name
        temp_dic["colWidth"] = 180
        table_columns.append(temp_dic)
    transposed_json = index_df.T.to_dict()
    table_data = []
    for key in transposed_json:
        table_data.append(transposed_json[key])
    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/cleanData", methods=["POST"])
def clean_data() -> Any:
    """
    This function apply clean functions on input dataset.
    """
    info = request.get_json()
    clean_func = info["clean_func"]
    col = info["col"]

    global index_df

    df_cleaned = clean_function_dic[clean_func](index_df, column=col, inplace=True)
    df_cleaned = df_cleaned.astype(str)
    col_names = df_cleaned.columns.values.tolist()
    table_columns = []
    for col_name in col_names:
        temp_dic = {}
        temp_dic["colName"] = col_name
        temp_dic["colLabel"] = col_name
        temp_dic["colWidth"] = 180
        table_columns.append(temp_dic)
    transposed_json = df_cleaned.T.to_dict()
    table_data = []
    for key in transposed_json:
        table_data.append(transposed_json[key])
    index_df = df_cleaned

    return {"tableData": table_data, "tableColumns": table_columns}


def launch(df) -> None:
    """
    This function incorporate the GUI of clean module into Jupyter Notebook.
    """
    global index_df
    index_df = df
    app_kwargs = {"port": 7680, "host": "localhost", "debug": False}
    thread = threading.Thread(target=app.run, kwargs=app_kwargs, daemon=True)
    thread.start()
