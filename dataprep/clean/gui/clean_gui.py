"""
Flask backend of Dataprep.Clean GUI.
"""
from typing import Any

import threading
import io
import copy
import time
import inspect
import logging

from base64 import b64encode
from os import path
from tempfile import mkdtemp
from ast import literal_eval

import pandas as pd

from IPython.display import Javascript, display

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

# disable useless log
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# pylint: disable=C0103, W0603, too-many-locals
origin_df = pd.DataFrame({})
index_df = pd.DataFrame({})

# List of operation log
all_logs = []
operation_log = []

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

DECODE_FUNC = """
    function b64DecodeUnicode(str) {
        // Going backwards: from bytestream, to percent-encoding, to original string.
        return decodeURIComponent(atob(str).split('').map(function(c) {
            return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
        }).join(''));
    }
"""

ts_list = []
ts_point = -1
tmp_dir = mkdtemp().replace("\\", "/")


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
    global origin_df

    if origin_df.empty:
        origin_df = index_df.copy(deep=True)
        global tmp_dir
        ts = time.time()
        df_file = path.join(tmp_dir, f"df_{str(int(ts))}.pkl").replace("\\", "/")
        origin_df.to_pickle(df_file)
        global ts_list
        global ts_point
        ts_list.append(ts)
        ts_point += 1

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
    # print(ts_list)
    # print(ts_point)

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/fileUpload", methods=["POST"])
def getUploadedFile() -> Any:
    """
    This function shows the uploaded table file in GUI.
    """
    file = request.files["file"]
    if not file:
        return "No File"
    stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    # csv_input = csv.reader(stream)
    # print(csv_input)
    # for row in csv_input:
    #    print(row)

    global index_df
    index_df = pd.read_csv(stream)
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


@app.route("/cleanWholeDF", methods=["POST"])
def cleanWholeDF() -> Any:
    """
    This function receive the request of clean whole dataframe.
    """
    params = request.form
    clean_headers = params["clean_headers"]
    data_type_detection = params["data_type_detection"]
    standardize_missing = params["standardize_missing_values"]
    downcast_memory = params["downcast_memory"]

    global index_df
    df_cleaned = clean_df(
        index_df,
        clean_header=clean_headers,
        data_type_detection=data_type_detection,
        standardize_missing_values=standardize_missing,
        downcast_memory=downcast_memory,
        report=False,
    )
    df_cleaned = df_cleaned[1].astype(str)
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

    global tmp_dir
    ts = time.time()
    df_file = path.join(tmp_dir, f"df_{str(int(ts))}.pkl").replace("\\", "/")
    index_df.to_pickle(df_file)
    global ts_list
    global ts_point
    ts_list.append(ts)
    ts_point += 1

    # Update log with clean_df function
    log = f"clean_df(df, clean_header={clean_headers},\
                         data_type_detection='{data_type_detection}',\
                         standardize_missing_values='{standardize_missing}',\
                         downcast_memory={downcast_memory})"
    global all_logs
    all_logs.append(log)
    global operation_log
    operation_log = all_logs[:ts_point]

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/getOperationLog", methods=["GET"])
def getOperationLog() -> Any:
    """
    This function receive the request of get log of operations.
    """
    global operation_log
    return {"operationLog": operation_log}


@app.route("/getFunctionParams", methods=["GET", "POST"])
def getFunctionParams() -> Any:
    """
    This function receive the request of get parameter of clean functions.
    """
    global index_df

    index_df = index_df.astype(str)
    col_names = index_df.columns.values.tolist()
    table_columns = []
    for col_name in col_names:
        temp_dic = {}
        temp_dic["value"] = col_name
        temp_dic["label"] = col_name
        table_columns.append(temp_dic)

    param_dic = {}
    param_default = {}
    info = request.get_json()
    clean_func = info["clean_func"]
    # print(clean_func)
    args = inspect.signature(clean_function_dic[clean_func]).parameters
    args_list = list(args.keys())
    for arg in args_list:
        temp_option_list = []
        if arg in ("df", "column"):
            continue
        if arg == "inplace":
            break
        if isinstance(args[arg].default, bool):
            temp_option_list.append({"value": "True", "label": "True"})
            temp_option_list.append({"value": "False", "label": "False"})
            param_dic[arg] = temp_option_list
            param_default[arg] = str(args[arg].default)
        if arg == "output_format":
            temp_option_list.append({"value": "standard", "label": "standard"})
            temp_option_list.append({"value": "compact", "label": "compact"})
            param_dic[arg] = temp_option_list
            param_default[arg] = str(args[arg].default)

    return {"tableColumns": table_columns, "paramDic": param_dic, "paramDefault": param_default}


@app.route("/cleanSingleCol", methods=["POST"])
def cleanSingleCol() -> Any:
    """
    This function receive the request of clean one single column.
    """
    params = request.form
    clean_func = params["clean_func"]
    selected_col = params["selected_col"]
    selected_params = literal_eval(params["selected_params"])
    param_str = ""
    for param_name in selected_params:
        if isinstance(literal_eval(selected_params[param_name]), str):
            param_str += f"{param_name}='{selected_params[param_name]}', "
        else:
            param_str += f"{param_name}={selected_params[param_name]}, "
        selected_params[param_name] = literal_eval(selected_params[param_name])
    param_str = param_str[: (len(param_str) - 2)]

    global index_df

    df_cleaned = clean_function_dic[clean_func](
        index_df, column=selected_col, report=False, **selected_params
    )
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

    global tmp_dir
    ts = time.time()
    df_file = path.join(tmp_dir, f"df_{str(int(ts))}.pkl").replace("\\", "/")
    index_df.to_pickle(df_file)
    global ts_list
    global ts_point
    ts_list.append(ts)
    ts_point += 1

    # Update log with clean_df function
    log = f"{clean_func}(df, column='{selected_col}', {param_str})"
    global all_logs
    all_logs.append(log)
    global operation_log
    operation_log = all_logs[:ts_point]

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/getOriginData", methods=["GET"])
def getOriginData() -> Any:
    """
    This function gets the origin data.
    """
    global index_df
    global origin_df
    # print(origin_df)
    index_df = copy.deepcopy(origin_df)

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

    global tmp_dir
    ts = time.time()
    df_file = path.join(tmp_dir, f"df_{str(int(ts))}.pkl").replace("\\", "/")
    index_df.to_pickle(df_file)
    global ts_list
    global ts_point
    ts_list = [ts]
    ts_point = 0

    global all_logs
    all_logs = []
    global operation_log
    operation_log = []

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/undo", methods=["GET"])
def getUndoData() -> Any:
    """
    This function undo one step operation.
    """
    global ts_point
    global ts_list
    if ts_point > 0:
        ts_point -= 1
    global tmp_dir
    df_file = path.join(tmp_dir, f"df_{str(int(ts_list[ts_point]))}.pkl").replace("\\", "/")

    global index_df
    index_df = pd.read_pickle(df_file)

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

    global operation_log
    operation_log = all_logs[:ts_point]

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/redo", methods=["GET"])
def getRedoData() -> Any:
    """
    This function redo the undoed operation.
    """

    global ts_point
    global ts_list
    if ts_point < len(ts_list) - 1:
        ts_point += 1
    global tmp_dir
    df_file = path.join(tmp_dir, f"df_{str(int(ts_list[ts_point]))}.pkl").replace("\\", "/")

    global index_df
    index_df = pd.read_pickle(df_file)

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

    global operation_log
    operation_log = all_logs[:ts_point]

    return {"tableData": table_data, "tableColumns": table_columns}


@app.route("/exportCSV", methods=["POST"])
def exportCSV() -> Any:
    """
    This function exports CSV.
    """
    global index_df
    index_df.to_csv("index_df.csv", index=False)

    f = open("index_df.csv", "rb")
    return f.read()


@app.route("/exportDF", methods=["GET"])
def exportDF() -> Any:
    """
    This function exports dataframe.
    """

    # def add_cell(text,  type='code', direct='above'):
    #    text = text.replace('\n','\\n').replace("\"", "\\\"").replace("'", "\\'")

    #    display(Javascript('''
    #    var cell = IPython.notebook.insert_cell_{}("{}")
    #    cell.set_text("{}")
    #    '''.format(direct, type, text)));

    # for i in range(3):
    #    add_cell(f'# heading{i}', 'markdown')
    #    add_cell(f'code {i}')

    global index_df

    code = "# dataframe with cleaned string values\ncleaned_df"
    encoded_code = (b64encode(str.encode(code))).decode()
    final_df = index_df.copy(deep=True)
    # create a temporary directory for the dataframe file
    tmp_dir = mkdtemp().replace("\\", "/")
    df_file = path.join(tmp_dir, "clean_gui_output.pkl").replace("\\", "/")
    final_df.to_pickle(df_file)
    # code to read the file and delete the temporary directory afterwards
    execute_code = (
        "import pandas as pd\n"
        "import shutil\n"
        f"cleaned_df = pd.read_pickle('{df_file}')\n"
        f"shutil.rmtree('{tmp_dir}')"
    )
    encoded_execute = (b64encode(str.encode(execute_code))).decode()
    code = f"""
             {DECODE_FUNC}
             IPython.notebook.kernel.execute(b64DecodeUnicode("{encoded_execute}"));
             var code = IPython.notebook.insert_cell_below('code');
             code.set_text(b64DecodeUnicode("{encoded_code}"));
             code.execute();
             """
    display(Javascript(code))
    return "success"


@app.route("/exportExecutionLog", methods=["GET"])
def exportExecutionLog() -> Any:
    """
    This function exports the log of execution.
    """
    notebook_type = "code"
    direct = "below"
    text = ""

    global operation_log
    for log in operation_log:
        text = text + "df = " + log + "\n"
    # print(text)
    text = text.replace("\n", "\\n").replace('"', '\\"').replace("'", "\\'")

    display(
        Javascript(
            """
        var cell = IPython.notebook.insert_cell_{}("{}")
        cell.set_text("{}")
        """.format(
                direct, notebook_type, text
            )
        )
    )

    return "success"


def launch(df) -> None:
    """
    This function incorporate the GUI of clean module into Jupyter Notebook.
    """
    global index_df
    index_df = df
    app_kwargs = {"port": 7680, "host": "localhost", "debug": False}
    thread = threading.Thread(target=app.run, kwargs=app_kwargs, daemon=True)
    thread.start()
