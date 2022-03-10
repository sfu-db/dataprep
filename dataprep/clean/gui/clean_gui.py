"""
Flask backend of Dataprep.Clean GUI.
"""
# pylint: disable=R0912, R0915
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
    clean_duplication,
    clean_url,
    clean_address,
    clean_df,
    clean_currency,
    clean_au_abn,
    clean_au_acn,
    clean_au_tfn,
    clean_be_iban,
    clean_be_vat,
    clean_bg_egn,
    clean_bg_pnf,
    clean_bg_vat,
    clean_br_cnpj,
    clean_br_cpf,
    clean_by_unp,
    clean_ca_bn,
    clean_ca_sin,
    clean_ch_esr,
    clean_ch_ssn,
    clean_ch_uid,
    clean_ch_vat,
    clean_cl_rut,
    clean_cn_ric,
    clean_cn_uscc,
    clean_co_nit,
    clean_cr_cpf,
    clean_cr_cpj,
    clean_cr_cr,
    clean_cu_ni,
    clean_cy_vat,
    clean_cz_dic,
    clean_cz_rc,
    clean_de_handelsregisternummer,
    clean_de_idnr,
    clean_de_stnr,
    clean_de_vat,
    clean_de_wkn,
    clean_dk_cpr,
    clean_dk_cvr,
    clean_do_cedula,
    clean_do_ncf,
    clean_do_rnc,
    clean_ec_ci,
    clean_ec_ruc,
    clean_ee_ik,
    clean_ee_kmkr,
    clean_ee_registrikood,
    clean_es_ccc,
    clean_es_cif,
    clean_es_cups,
    clean_es_dni,
    clean_es_iban,
    clean_es_nie,
    clean_es_nif,
    clean_es_referenciacatastral,
    clean_eu_at_02,
    clean_eu_banknote,
    clean_eu_eic,
    clean_eu_nace,
    clean_eu_vat,
    clean_fi_alv,
    clean_fi_associationid,
    clean_fi_hetu,
    clean_fi_veronumero,
    clean_fi_ytunnus,
    clean_fr_nif,
    clean_fr_nir,
    clean_fr_siren,
    clean_fr_siret,
    clean_fr_tva,
    clean_gb_nhs,
    clean_gb_sedol,
    clean_gb_upn,
    clean_gb_utr,
    clean_gb_vat,
    clean_gr_amka,
    clean_gr_vat,
    clean_gt_nit,
    clean_hr_oib,
    clean_hu_anum,
    clean_id_npwp,
    clean_ie_pps,
    clean_ie_vat,
    clean_il_hp,
    clean_il_idnr,
    clean_in_aadhaar,
    clean_in_pan,
    clean_is_kennitala,
    clean_is_vsk,
    clean_it_aic,
    clean_it_codicefiscale,
    clean_it_iva,
    clean_jp_cn,
    clean_kr_brn,
    clean_kr_rrn,
    clean_li_peid,
    clean_lt_pvm,
    clean_lt_asmens,
    clean_lu_tva,
    clean_lv_pvn,
    clean_mc_tva,
    clean_md_idno,
    clean_me_iban,
    clean_mt_vat,
    clean_mu_nid,
    clean_mx_curp,
    clean_mx_rfc,
    clean_my_nric,
    clean_nl_brin,
    clean_nl_btw,
    clean_nl_bsn,
    clean_nl_onderwijsnummer,
    clean_nl_postcode,
    clean_no_fodselsnummer,
    clean_no_iban,
    clean_no_kontonr,
    clean_no_mva,
    clean_no_orgnr,
    clean_nz_bankaccount,
    clean_nz_ird,
    clean_pe_cui,
    clean_pe_ruc,
    clean_pl_nip,
    clean_pl_pesel,
    clean_pl_regon,
    clean_pt_nif,
    clean_py_ruc,
    clean_ro_cf,
    clean_ro_cnp,
    clean_ro_cui,
    clean_ro_onrc,
    clean_isbn,
    clean_bic,
    clean_bitcoin,
    clean_casrn,
    clean_cusip,
    clean_ean,
    clean_figi,
    clean_grid,
    clean_iban,
    clean_imei,
    clean_imo,
    clean_imsi,
    clean_isan,
    clean_isil,
    clean_isin,
    clean_ismn,
    clean_issn,
    clean_ad_nrt,
    clean_al_nipt,
    clean_ar_cbu,
    clean_ar_cuit,
    clean_ar_dni,
    clean_at_uid,
    clean_at_vnr,
    clean_lei,
    clean_meid,
    clean_vatin,
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
    "clean_url": clean_url,
    "clean_address": clean_address,
    "clean_df": clean_df,
    # Uncomment when add the function to backend
    # "clean_duplication": clean_duplication,
    # "clean_currency": clean_currency,
    # "clean_au_abn": clean_au_abn,
    # "clean_au_acn": clean_au_acn,
    # "clean_au_tfn": clean_au_tfn,
    # "clean_be_iban": clean_be_iban,
    # "clean_be_vat": clean_be_vat,
    # "clean_bg_egn": clean_bg_egn,
    # "clean_bg_pnf": clean_bg_pnf,
    # "clean_bg_vat": clean_bg_vat,
    # "clean_br_cnpj": clean_br_cnpj,
    # "clean_br_cpf": clean_br_cpf,
    # "clean_by_unp": clean_by_unp,
    # "clean_ca_bn": clean_ca_bn,
    # "clean_ca_sin": clean_ca_sin,
    # "clean_ch_esr": clean_ch_esr,
    # "clean_ch_ssn": clean_ch_ssn,
    # "clean_ch_uid": clean_ch_uid,
    # "clean_ch_vat": clean_ch_vat,
    # "clean_cl_rut": clean_cl_rut,
    # "clean_cn_ric": clean_cn_ric,
    # "clean_cn_uscc": clean_cn_uscc,
    # "clean_co_nit": clean_co_nit,
    # "clean_cr_cpf": clean_cr_cpf,
    # "clean_cr_cpj": clean_cr_cpj,
    # "clean_cr_cr": clean_cr_cr,
    # "clean_cu_ni": clean_cu_ni,
    # "clean_cy_vat": clean_cy_vat,
    # "clean_cz_dic": clean_cz_dic,
    # "clean_cz_rc": clean_cz_rc,
    # "clean_de_handelsregisternummer": clean_de_handelsregisternummer,
    # "clean_de_idnr": clean_de_idnr,
    # "clean_de_stnr": clean_de_stnr,
    # "clean_de_vat": clean_de_vat,
    # "clean_de_wkn": clean_de_wkn,
    # "clean_dk_cpr": clean_dk_cpr,
    # "clean_dk_cvr": clean_dk_cvr,
    # "clean_do_cedula": clean_do_cedula,
    # "clean_do_ncf": clean_do_ncf,
    # "clean_do_rnc": clean_do_rnc,
    # "clean_ec_ci": clean_ec_ci,
    # "clean_ec_ruc": clean_ec_ruc,
    # "clean_ee_ik": clean_ee_ik,
    # "clean_ee_kmkr": clean_ee_kmkr,
    # "clean_ee_registrikood": clean_ee_registrikood,
    # "clean_es_ccc": clean_es_ccc,
    # "clean_es_cif": clean_es_cif,
    # "clean_es_cups": clean_es_cups,
    # "clean_es_dni": clean_es_dni,
    # "clean_es_iban": clean_es_iban,
    # "clean_es_nie": clean_es_nie,
    # "clean_es_nif": clean_es_nif,
    # "clean_es_referenciacatastral": clean_es_referenciacatastral,
    # "clean_eu_at_02": clean_eu_at_02,
    # "clean_eu_banknote": clean_eu_banknote,
    # "clean_eu_eic": clean_eu_eic,
    # "clean_eu_nace": clean_eu_nace,
    # "clean_eu_vat": clean_eu_vat,
    # "clean_fi_alv": clean_fi_alv,
    # "clean_fi_associationid": clean_fi_associationid,
    # "clean_fi_hetu": clean_fi_hetu,
    # "clean_fi_veronumero": clean_fi_veronumero,
    # "clean_fi_ytunnus": clean_fi_ytunnus,
    # "clean_fr_nif": clean_fr_nif,
    # "clean_fr_nir": clean_fr_nir,
    # "clean_fr_siren": clean_fr_siren,
    # "clean_fr_siret": clean_fr_siret,
    # "clean_fr_tva": clean_fr_tva,
    # "clean_gb_nhs": clean_gb_nhs,
    # "clean_gb_sedol": clean_gb_sedol,
    # "clean_gb_upn": clean_gb_upn,
    # "clean_gb_utr": clean_gb_utr,
    # "clean_gb_vat": clean_gb_vat,
    # "clean_gr_amka": clean_gr_amka,
    # "clean_gr_vat": clean_gr_vat,
    # "clean_gt_nit": clean_gt_nit,
    # "clean_hr_oib": clean_hr_oib,
    # "clean_hu_anum": clean_hu_anum,
    # "clean_id_npwp": clean_id_npwp,
    # "clean_ie_pps": clean_ie_pps,
    # "clean_ie_vat": clean_ie_vat,
    # "clean_il_hp": clean_il_hp,
    # "clean_il_idnr": clean_il_idnr,
    # "clean_in_aadhaar": clean_in_aadhaar,
    # "clean_in_pan": clean_in_pan,
    # "clean_is_kennitala": clean_is_kennitala,
    # "clean_is_vsk": clean_is_vsk,
    # "clean_it_aic": clean_it_aic,
    # "clean_it_codicefiscale": clean_it_codicefiscale,
    # "clean_it_iva": clean_it_iva,
    # "clean_jp_cn": clean_jp_cn,
    # "clean_kr_brn": clean_kr_brn,
    # "clean_kr_rrn": clean_kr_rrn,
    # "clean_li_peid": clean_li_peid,
    # "clean_lt_pvm": clean_lt_pvm,
    # "clean_lt_asmens": clean_lt_asmens,
    # "clean_lu_tva": clean_lu_tva,
    # "clean_lv_pvn": clean_lv_pvn,
    # "clean_mc_tva": clean_mc_tva,
    # "clean_md_idno": clean_md_idno,
    # "clean_me_iban": clean_me_iban,
    # "clean_mt_vat": clean_mt_vat,
    # "clean_mu_nid": clean_mu_nid,
    # "clean_mx_curp": clean_mx_curp,
    # "clean_mx_rfc": clean_mx_rfc,
    # "clean_my_nric": clean_my_nric,
    # "clean_nl_brin": clean_nl_brin,
    # "clean_nl_btw": clean_nl_btw,
    # "clean_nl_bsn": clean_nl_bsn,
    # "clean_nl_onderwijsnummer": clean_nl_onderwijsnummer,
    # "clean_nl_postcode": clean_nl_postcode,
    # "clean_no_fodselsnummer": clean_no_fodselsnummer,
    # "clean_no_iban": clean_no_iban,
    # "clean_no_kontonr": clean_no_kontonr,
    # "clean_no_mva": clean_no_mva,
    # "clean_no_orgnr": clean_no_orgnr,
    # "clean_nz_bankaccount": clean_nz_bankaccount,
    # "clean_nz_ird": clean_nz_ird,
    # "clean_pe_cui": clean_pe_cui,
    # "clean_pe_ruc": clean_pe_ruc,
    # "clean_pl_nip": clean_pl_nip,
    # "clean_pl_pesel": clean_pl_pesel,
    # "clean_pl_regon": clean_pl_regon,
    # "clean_pt_nif": clean_pt_nif,
    # "clean_py_ruc": clean_py_ruc,
    # "clean_ro_cf": clean_ro_cf,
    # "clean_ro_cnp": clean_ro_cnp,
    # "clean_ro_cui": clean_ro_cui,
    # "clean_ro_onrc": clean_ro_onrc,
    # "clean_isbn": clean_isbn,
    # "clean_bic":clean_bic,
    # "clean_bitcoin":clean_bitcoin,
    # "clean_casrn":clean_casrn,
    # "clean_cusip":clean_cusip,
    # "clean_ean":clean_ean,
    # "clean_figi":clean_figi,
    # "clean_grid":clean_grid,
    # "clean_iban":clean_iban,
    # "clean_imei":clean_imei,
    # "clean_imo":clean_imo,
    # "clean_imsi":clean_imsi,
    # "clean_isan":clean_isan,
    # "clean_isil":clean_isil,
    # "clean_isin":clean_isin,
    # "clean_ismn":clean_ismn,
    # "clean_issn":clean_issn,
    # "clean_ad_nrt":clean_ad_nrt,
    # "clean_al_nipt":clean_al_nipt,
    # "clean_ar_cbu":clean_ar_cbu,
    # "clean_ar_cuit":clean_ar_cuit,
    # "clean_ar_dni":clean_ar_dni,
    # "clean_at_uid":clean_at_uid,
    # "clean_at_vnr":clean_at_vnr,
    # "clean_lei":clean_lei,
    # "clean_meid":clean_meid,
    # "clean_vatin":clean_vatin,
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
    log = (
        f"clean_df(df, clean_header={clean_headers},"
        f"data_type_detection='{data_type_detection}',"
        f"standardize_missing_values='{standardize_missing}',"
        f"downcast_memory={downcast_memory})"
    )
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

    if clean_func == "clean_email":
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
    elif clean_func == "clean_headers":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())

        for arg in args_list:
            temp_option_list = []
            if arg == "df":
                continue
            if arg == "case":
                temp_option_list.append({"value": "snake", "label": "Snake ('column_name')"})
                temp_option_list.append({"value": "kebab", "label": "Kebab ('column-name')"})
                temp_option_list.append({"value": "camel", "label": "Camel ('columnName')"})
                temp_option_list.append({"value": "pascal", "label": "Pascal ('ColumnName')"})
                temp_option_list.append({"value": "const", "label": "Const ('COLUMN_NAME')"})
                temp_option_list.append({"value": "sentence", "label": "Sentence ('Column name')"})
                temp_option_list.append({"value": "title", "label": "Title ('Column Name')"})
                temp_option_list.append({"value": "lower", "label": "Lower ('column name')"})
                temp_option_list.append({"value": "upper", "label": "Upper ('COLUMN NAME')"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "remove_accents":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_country":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())
        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "input_format":
                temp_option_list.append({"value": "auto", "label": "Auto"})
                temp_option_list.append({"value": "name", "label": "Name"})
                temp_option_list.append({"value": "official", "label": "Official"})
                temp_option_list.append({"value": "alpha-2", "label": "Alpha-2"})
                temp_option_list.append({"value": "alpha-3", "label": "Alpha-3"})
                temp_option_list.append({"value": "numeric", "label": "Numeric"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append({"value": "name", "label": "Name"})
                temp_option_list.append({"value": "official", "label": "Official"})
                temp_option_list.append({"value": "alpha-2", "label": "Alpha-2"})
                temp_option_list.append({"value": "alpha-3", "label": "Alpha-3"})
                temp_option_list.append({"value": "numeric", "label": "Numeric"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fuzzy_dist":
                for i in range(0, 11):
                    temp_option_list.append({"value": str(i), "label": str(i)})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "strict":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_date":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())

        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "output_format":
                temp_option_list.append(
                    {"value": "YYYY-MM-DD hh:mm:ss", "label": "YYYY-MM-DD hh:mm:ss"}
                )
                temp_option_list.append({"value": "YYYY-MM-DD", "label": "YYYY-MM-DD"})
                temp_option_list.append(
                    {"value": "YYYY-MM-DD AD at hh:mm:ss Z", "label": "YYYY-MM-DD AD at hh:mm:ss Z"}
                )
                temp_option_list.append(
                    {"value": "EEE, d MMM yyyy HH:mm:ss Z", "label": "EEE, d MMM yyyy HH:mm:ss Z"}
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fix_missing":
                temp_option_list.append(
                    {
                        "value": "minimum",
                        "label": "Fill hour, minute, second with 0, "
                        "and month, day, year with January, 1st, 2000.",
                    }
                )
                temp_option_list.append(
                    {"value": "current", "label": "Fill with the current date and time"}
                )
                temp_option_list.append({"value": "empty", "label": "None"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "infer_day_first":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_lat_long":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())
        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append(
                    {"value": "dd", "label": "Decimal degrees (51.4934, 0.0098)"}
                )
                temp_option_list.append(
                    {
                        "value": "ddh",
                        "label": "Decimal degrees with hemisphere ('51.4934° N, 0.0098° E')",
                    }
                )
                temp_option_list.append(
                    {"value": "dm", "label": "Degrees minutes ('51° 29.604' N, 0° 0.588' E')"}
                )
                temp_option_list.append(
                    {
                        "value": "dms",
                        "label": "Degrees minutes seconds ('51° 29' 36.24\" N, 0° 0' 35.28\" E')",
                    }
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_ip":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())
        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "input_format":
                temp_option_list.append({"value": "auto", "label": "Auto"})
                temp_option_list.append({"value": "ipv4", "label": "IPV4"})
                temp_option_list.append({"value": "ipv6", "label": "IPV6"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "output_format":
                temp_option_list.append({"value": "compressed", "label": "Compressed (12.3.4.5)"})
                temp_option_list.append({"value": "full", "label": "Full ('0012.0003.0004.0005')"})
                temp_option_list.append(
                    {"value": "binary", "label": "Binary ('00001100000000110000010000000101')"}
                )
                temp_option_list.append({"value": "hexa", "label": "Hexa ('0xc030405')"})
                temp_option_list.append({"value": "integer", "label": "Integer (201524229)"})
                temp_option_list.append({"value": "packed", "label": "Packed"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_phone":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())

        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "output_format":
                temp_option_list.append({"value": "nanp", "label": "nanp ('NPA-NXX-XXXX')"})
                temp_option_list.append({"value": "e164", "label": "e164 ('+1NPANXXXXXX')"})
                temp_option_list.append(
                    {"value": "national", "label": "National ('(NPA) NXX-XXXX)"}
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "fix_missing":
                temp_option_list.append({"value": "empty", "label": "Empty"})
                temp_option_list.append({"value": "auto", "label": "Auto"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    # elif clean_func == "clean_text":
    #    pass
    elif clean_func == "clean_url":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())
        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg in ["remove_auth", "split"]:
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    elif clean_func == "clean_address":
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())

        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
            if arg == "output_format":
                temp_option_list.append(
                    {
                        "value": "(building) house_number street_prefix_abbr street_name "
                        "street_suffix_abbr,apartment, city, state_abbr zipcode",
                        "label": "(building) house_number street_prefix_abbr street_name "
                        "street_suffix_abbr,apartment, city, state_abbr zipcode",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "(zipcode) street_prefix_full street_name ~state_full~",
                        "label": "(zipcode) street_prefix_full street_name ~state_full~",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "house_number street_name street_suffix_full (building)",
                        "label": "house_number street_name street_suffix_full (building)",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
    else:
        args = inspect.signature(clean_function_dic[clean_func]).parameters
        args_list = list(args.keys())
        for arg in args_list:
            temp_option_list = []
            if arg in ("df", "column"):
                continue
            if arg == "inplace":
                break
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
    df_cleaned = ""
    global index_df

    try:
        for param_name in selected_params:
            if clean_func in [
                "clean_email",
                "clean_country",
                "clean_ip",
                "clean_phone",
                "clean_url",
                "clean_headers",
                "clean_date",
                "clean_lat_long",
                # "clean_text",
                "clean_address",
                "clean_df",
                # Uncomment when add the function to backend
                # "clean_duplication",
                # "clean_currency",
                # "clean_au_abn",
                # "clean_au_acn",
                # "clean_au_tfn",
                # "clean_be_iban",
                # "clean_be_vat",
                # "clean_bg_egn",
                # "clean_bg_pnf",
                # "clean_bg_vat",
                # "clean_br_cnpj",
                # "clean_br_cpf",
                # "clean_by_unp",
                # "clean_ca_bn",
                # "clean_ca_sin",
                # "clean_ch_esr",
                # "clean_ch_ssn",
                # "clean_ch_uid",
                # "clean_ch_vat",
                # "clean_cl_rut",
                # "clean_cn_ric",
                # "clean_cn_uscc",
                # "clean_co_nit",
                # "clean_cr_cpf",
                # "clean_cr_cpj",
                # "clean_cr_cr",
                # "clean_cu_ni",
                # "clean_cy_vat",
                # "clean_cz_dic",
                # "clean_cz_rc",
                # "clean_de_handelsregisternummer",
                # "clean_de_idnr",
                # "clean_de_stnr",
                # "clean_de_vat",
                # "clean_de_wkn",
                # "clean_dk_cpr",
                # "clean_dk_cvr",
                # "clean_do_cedula",
                # "clean_do_ncf",
                # "clean_do_rnc",
                # "clean_ec_ci",
                # "clean_ec_ruc",
                # "clean_ee_ik",
                # "clean_ee_kmkr",
                # "clean_ee_registrikood",
                # "clean_es_ccc",
                # "clean_es_cif",
                # "clean_es_cups",
                # "clean_es_dni",
                # "clean_es_iban",
                # "clean_es_nie",
                # "clean_es_nif",
                # "clean_es_referenciacatastral",
                # "clean_eu_at_02",
                # "clean_eu_banknote",
                # "clean_eu_eic",
                # "clean_eu_nace",
                # "clean_eu_vat",
                # "clean_fi_alv",
                # "clean_fi_associationid",
                # "clean_fi_hetu",
                # "clean_fi_veronumero",
                # "clean_fi_ytunnus",
                # "clean_fr_nif",
                # "clean_fr_nir",
                # "clean_fr_siren",
                # "clean_fr_siret",
                # "clean_fr_tva",
                # "clean_gb_nhs",
                # "clean_gb_sedol",
                # "clean_gb_upn",
                # "clean_gb_utr",
                # "clean_gb_vat",
                # "clean_gr_amka",
                # "clean_gr_vat",
                # "clean_gt_nit",
                # "clean_hr_oib",
                # "clean_hu_anum",
                # "clean_id_npwp",
                # "clean_ie_pps",
                # "clean_ie_vat",
                # "clean_il_hp",
                # "clean_il_idnr",
                # "clean_in_aadhaar",
                # "clean_in_pan",
                # "clean_is_kennitala",
                # "clean_is_vsk",
                # "clean_it_aic",
                # "clean_it_codicefiscale",
                # "clean_it_iva",
                # "clean_jp_cn",
                # "clean_kr_brn",
                # "clean_kr_rrn",
                # "clean_li_peid",
                # "clean_lt_pvm",
                # "clean_lt_asmens",
                # "clean_lu_tva",
                # "clean_lv_pvn",
                # "clean_mc_tva",
                # "clean_md_idno",
                # "clean_me_iban",
                # "clean_mt_vat",
                # "clean_mu_nid",
                # "clean_mx_curp",
                # "clean_mx_rfc",
                # "clean_my_nric",
                # "clean_nl_brin",
                # "clean_nl_btw",
                # "clean_nl_bsn",
                # "clean_nl_onderwijsnummer",
                # "clean_nl_postcode",
                # "clean_no_fodselsnummer",
                # "clean_no_iban",
                # "clean_no_kontonr",
                # "clean_no_mva",
                # "clean_no_orgnr",
                # "clean_nz_bankaccount",
                # "clean_nz_ird",
                # "clean_pe_cui",
                # "clean_pe_ruc",
                # "clean_pl_nip",
                # "clean_pl_pesel",
                # "clean_pl_regon",
                # "clean_pt_nif",
                # "clean_py_ruc",
                # "clean_ro_cf",
                # "clean_ro_cnp",
                # "clean_ro_cui",
                # "clean_ro_onrc",
                # "clean_isbn",
            ]:
                if selected_params[param_name] == "True":
                    param_str += f"{param_name}={selected_params[param_name]}, "
                    selected_params[param_name] = True
                elif selected_params[param_name] == "False":
                    param_str += f"{param_name}={selected_params[param_name]}, "
                    selected_params[param_name] = False
                elif selected_params[param_name].isnumeric():
                    param_str += f"{param_name}={selected_params[param_name]}, "
                    selected_params[param_name] = int(selected_params[param_name])
                else:
                    param_str += f"{param_name}='{selected_params[param_name]}', "
            else:
                param_str += f"{param_name}={selected_params[param_name]}, "
                selected_params[param_name] = selected_params[param_name]

        param_str = param_str[: (len(param_str) - 2)]

        if clean_func in [
            "clean_email",
            "clean_country",
            "clean_ip",
            "clean_phone",
            "clean_url",
            "clean_date",
            "clean_address",
            # Uncomment when add the function to backend
            # "clean_duplication",
            # "clean_currency",
            # "clean_au_abn",
            # "clean_au_acn",
            # "clean_au_tfn",
            # "clean_be_iban",
            # "clean_be_vat",
            # "clean_bg_egn",
            # "clean_bg_pnf",
            # "clean_bg_vat",
            # "clean_br_cnpj",
            # "clean_br_cpf",
            # "clean_by_unp",
            # "clean_ca_bn",
            # "clean_ca_sin",
            # "clean_ch_esr",
            # "clean_ch_ssn",
            # "clean_ch_uid",
            # "clean_ch_vat",
            # "clean_cl_rut",
            # "clean_cn_ric",
            # "clean_cn_uscc",
            # "clean_co_nit",
            # "clean_cr_cpf",
            # "clean_cr_cpj",
            # "clean_cr_cr",
            # "clean_cu_ni",
            # "clean_cy_vat",
            # "clean_cz_dic",
            # "clean_cz_rc",
            # "clean_de_handelsregisternummer",
            # "clean_de_idnr",
            # "clean_de_stnr",
            # "clean_de_vat",
            # "clean_de_wkn",
            # "clean_dk_cpr",
            # "clean_dk_cvr",
            # "clean_do_cedula",
            # "clean_do_ncf",
            # "clean_do_rnc",
            # "clean_ec_ci",
            # "clean_ec_ruc",
            # "clean_ee_ik",
            # "clean_ee_kmkr",
            # "clean_ee_registrikood",
            # "clean_es_ccc",
            # "clean_es_cif",
            # "clean_es_cups",
            # "clean_es_dni",
            # "clean_es_iban",
            # "clean_es_nie",
            # "clean_es_nif",
            # "clean_es_referenciacatastral",
            # "clean_eu_at_02",
            # "clean_eu_banknote",
            # "clean_eu_eic",
            # "clean_eu_nace",
            # "clean_eu_vat",
            # "clean_fi_alv",
            # "clean_fi_associationid",
            # "clean_fi_hetu",
            # "clean_fi_veronumero",
            # "clean_fi_ytunnus",
            # "clean_fr_nif",
            # "clean_fr_nir",
            # "clean_fr_siren",
            # "clean_fr_siret",
            # "clean_fr_tva",
            # "clean_gb_nhs",
            # "clean_gb_sedol",
            # "clean_gb_upn",
            # "clean_gb_utr",
            # "clean_gb_vat",
            # "clean_gr_amka",
            # "clean_gr_vat",
            # "clean_gt_nit",
            # "clean_hr_oib",
            # "clean_hu_anum",
            # "clean_id_npwp",
            # "clean_ie_pps",
            # "clean_ie_vat",
            # "clean_il_hp",
            # "clean_il_idnr",
            # "clean_in_aadhaar",
            # "clean_in_pan",
            # "clean_is_kennitala",
            # "clean_is_vsk",
            # "clean_it_aic",
            # "clean_it_codicefiscale",
            # "clean_it_iva",
            # "clean_jp_cn",
            # "clean_kr_brn",
            # "clean_kr_rrn",
            # "clean_li_peid",
            # "clean_lt_pvm",
            # "clean_lt_asmens",
            # "clean_lu_tva",
            # "clean_lv_pvn",
            # "clean_mc_tva",
            # "clean_md_idno",
            # "clean_me_iban",
            # "clean_mt_vat",
            # "clean_mu_nid",
            # "clean_mx_curp",
            # "clean_mx_rfc",
            # "clean_my_nric",
            # "clean_nl_brin",
            # "clean_nl_btw",
            # "clean_nl_bsn",
            # "clean_nl_onderwijsnummer",
            # "clean_nl_postcode",
            # "clean_no_fodselsnummer",
            # "clean_no_iban",
            # "clean_no_kontonr",
            # "clean_no_mva",
            # "clean_no_orgnr",
            # "clean_nz_bankaccount",
            # "clean_nz_ird",
            # "clean_pe_cui",
            # "clean_pe_ruc",
            # "clean_pl_nip",
            # "clean_pl_pesel",
            # "clean_pl_regon",
            # "clean_pt_nif",
            # "clean_py_ruc",
            # "clean_ro_cf",
            # "clean_ro_cnp",
            # "clean_ro_cui",
            # "clean_ro_onrc",
            # "clean_isbn",
        ]:
            df_cleaned = clean_function_dic[clean_func](
                index_df, column=selected_col, report=False, **selected_params
            )
        elif clean_func == "clean_lat_long":
            df_cleaned = clean_function_dic[clean_func](
                index_df, lat_long=selected_col, report=False, **selected_params
            )
            # elif clean_func == "clean_text":
            #    df_cleaned = clean_function_dic[clean_func](
            #        index_df, column=selected_col
            #    )
        elif clean_func == "clean_headers":
            df_cleaned = clean_function_dic[clean_func](index_df, **selected_params)
        # elif clean_func in "clean_headers" or clean_func in "clean_lat_long":
        #    df_cleaned = clean_function_dic[clean_func](index_df, **selected_params)
        else:
            df_cleaned = clean_function_dic[clean_func](
                index_df, column=selected_col, **selected_params
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

    except TypeError:
        print(f'{"There is a type error."}')
    except KeyError:
        print(f"The {selected_col} column can not clean, please try it again.")
    except ValueError:
        print(f'{"Some field value is not valid"}')


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
