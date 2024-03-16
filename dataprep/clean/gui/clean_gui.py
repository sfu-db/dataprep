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
    clean_json,
    clean_url,
    clean_address,
    clean_df,
    clean_currency,
)
from dataprep.clean.clean_au_abn import clean_au_abn
from dataprep.clean.clean_au_acn import clean_au_acn
from dataprep.clean.clean_au_tfn import clean_au_tfn
from dataprep.clean.clean_be_iban import clean_be_iban
from dataprep.clean.clean_be_vat import clean_be_vat
from dataprep.clean.clean_bg_egn import clean_bg_egn
from dataprep.clean.clean_bg_pnf import clean_bg_pnf
from dataprep.clean.clean_bg_vat import clean_bg_vat
from dataprep.clean.clean_br_cnpj import clean_br_cnpj
from dataprep.clean.clean_br_cpf import clean_br_cpf
from dataprep.clean.clean_by_unp import clean_by_unp
from dataprep.clean.clean_ca_bn import clean_ca_bn
from dataprep.clean.clean_ca_sin import clean_ca_sin
from dataprep.clean.clean_ch_esr import clean_ch_esr
from dataprep.clean.clean_ch_ssn import clean_ch_ssn
from dataprep.clean.clean_ch_uid import clean_ch_uid
from dataprep.clean.clean_ch_vat import clean_ch_vat
from dataprep.clean.clean_cl_rut import clean_cl_rut
from dataprep.clean.clean_cn_ric import clean_cn_ric
from dataprep.clean.clean_cn_uscc import clean_cn_uscc
from dataprep.clean.clean_co_nit import clean_co_nit
from dataprep.clean.clean_cr_cpf import clean_cr_cpf
from dataprep.clean.clean_cr_cpj import clean_cr_cpj
from dataprep.clean.clean_cr_cr import clean_cr_cr
from dataprep.clean.clean_cu_ni import clean_cu_ni
from dataprep.clean.clean_cy_vat import clean_cy_vat
from dataprep.clean.clean_cz_dic import clean_cz_dic
from dataprep.clean.clean_cz_rc import clean_cz_rc
from dataprep.clean.clean_de_handelsregisternummer import clean_de_handelsregisternummer
from dataprep.clean.clean_de_idnr import clean_de_idnr
from dataprep.clean.clean_de_stnr import clean_de_stnr
from dataprep.clean.clean_de_vat import clean_de_vat
from dataprep.clean.clean_de_wkn import clean_de_wkn
from dataprep.clean.clean_dk_cpr import clean_dk_cpr
from dataprep.clean.clean_dk_cvr import clean_dk_cvr
from dataprep.clean.clean_do_cedula import clean_do_cedula
from dataprep.clean.clean_do_ncf import clean_do_ncf
from dataprep.clean.clean_do_rnc import clean_do_rnc
from dataprep.clean.clean_ec_ci import clean_ec_ci
from dataprep.clean.clean_ec_ruc import clean_ec_ruc
from dataprep.clean.clean_ee_ik import clean_ee_ik
from dataprep.clean.clean_ee_kmkr import clean_ee_kmkr
from dataprep.clean.clean_ee_registrikood import clean_ee_registrikood
from dataprep.clean.clean_es_ccc import clean_es_ccc
from dataprep.clean.clean_es_cif import clean_es_cif
from dataprep.clean.clean_es_cups import clean_es_cups
from dataprep.clean.clean_es_dni import clean_es_dni
from dataprep.clean.clean_es_iban import clean_es_iban
from dataprep.clean.clean_es_nie import clean_es_nie
from dataprep.clean.clean_es_nif import clean_es_nif
from dataprep.clean.clean_es_referenciacatastral import clean_es_referenciacatastral
from dataprep.clean.clean_eu_at_02 import clean_eu_at_02
from dataprep.clean.clean_eu_banknote import clean_eu_banknote
from dataprep.clean.clean_eu_eic import clean_eu_eic
from dataprep.clean.clean_eu_nace import clean_eu_nace
from dataprep.clean.clean_eu_vat import clean_eu_vat
from dataprep.clean.clean_fi_alv import clean_fi_alv
from dataprep.clean.clean_fi_associationid import clean_fi_associationid
from dataprep.clean.clean_fi_hetu import clean_fi_hetu
from dataprep.clean.clean_fi_veronumero import clean_fi_veronumero
from dataprep.clean.clean_fi_ytunnus import clean_fi_ytunnus
from dataprep.clean.clean_fr_nif import clean_fr_nif
from dataprep.clean.clean_fr_nir import clean_fr_nir
from dataprep.clean.clean_fr_siren import clean_fr_siren
from dataprep.clean.clean_fr_siret import clean_fr_siret
from dataprep.clean.clean_fr_tva import clean_fr_tva
from dataprep.clean.clean_gb_nhs import clean_gb_nhs
from dataprep.clean.clean_gb_sedol import clean_gb_sedol
from dataprep.clean.clean_gb_upn import clean_gb_upn
from dataprep.clean.clean_gb_utr import clean_gb_utr
from dataprep.clean.clean_gb_vat import clean_gb_vat
from dataprep.clean.clean_gr_amka import clean_gr_amka
from dataprep.clean.clean_gr_vat import clean_gr_vat
from dataprep.clean.clean_gt_nit import clean_gt_nit
from dataprep.clean.clean_hr_oib import clean_hr_oib
from dataprep.clean.clean_hu_anum import clean_hu_anum
from dataprep.clean.clean_id_npwp import clean_id_npwp
from dataprep.clean.clean_ie_pps import clean_ie_pps
from dataprep.clean.clean_ie_vat import clean_ie_vat
from dataprep.clean.clean_il_hp import clean_il_hp
from dataprep.clean.clean_il_idnr import clean_il_idnr
from dataprep.clean.clean_in_aadhaar import clean_in_aadhaar
from dataprep.clean.clean_in_pan import clean_in_pan
from dataprep.clean.clean_is_kennitala import clean_is_kennitala
from dataprep.clean.clean_is_vsk import clean_is_vsk
from dataprep.clean.clean_it_aic import clean_it_aic
from dataprep.clean.clean_it_codicefiscale import clean_it_codicefiscale
from dataprep.clean.clean_it_iva import clean_it_iva
from dataprep.clean.clean_jp_cn import clean_jp_cn
from dataprep.clean.clean_kr_brn import clean_kr_brn
from dataprep.clean.clean_kr_rrn import clean_kr_rrn
from dataprep.clean.clean_li_peid import clean_li_peid
from dataprep.clean.clean_lt_pvm import clean_lt_pvm
from dataprep.clean.clean_lt_asmens import clean_lt_asmens
from dataprep.clean.clean_lu_tva import clean_lu_tva
from dataprep.clean.clean_lv_pvn import clean_lv_pvn
from dataprep.clean.clean_mc_tva import clean_mc_tva
from dataprep.clean.clean_md_idno import clean_md_idno
from dataprep.clean.clean_me_iban import clean_me_iban
from dataprep.clean.clean_mt_vat import clean_mt_vat
from dataprep.clean.clean_mu_nid import clean_mu_nid
from dataprep.clean.clean_mx_curp import clean_mx_curp
from dataprep.clean.clean_mx_rfc import clean_mx_rfc
from dataprep.clean.clean_my_nric import clean_my_nric
from dataprep.clean.clean_nl_brin import clean_nl_brin
from dataprep.clean.clean_nl_btw import clean_nl_btw
from dataprep.clean.clean_nl_bsn import clean_nl_bsn
from dataprep.clean.clean_nl_onderwijsnummer import clean_nl_onderwijsnummer
from dataprep.clean.clean_nl_postcode import clean_nl_postcode
from dataprep.clean.clean_no_fodselsnummer import clean_no_fodselsnummer
from dataprep.clean.clean_no_iban import clean_no_iban
from dataprep.clean.clean_no_kontonr import clean_no_kontonr
from dataprep.clean.clean_no_mva import clean_no_mva
from dataprep.clean.clean_no_orgnr import clean_no_orgnr
from dataprep.clean.clean_nz_bankaccount import clean_nz_bankaccount
from dataprep.clean.clean_nz_ird import clean_nz_ird
from dataprep.clean.clean_pe_cui import clean_pe_cui
from dataprep.clean.clean_pe_ruc import clean_pe_ruc
from dataprep.clean.clean_pl_nip import clean_pl_nip
from dataprep.clean.clean_pl_pesel import clean_pl_pesel
from dataprep.clean.clean_pl_regon import clean_pl_regon
from dataprep.clean.clean_pt_nif import clean_pt_nif
from dataprep.clean.clean_py_ruc import clean_py_ruc
from dataprep.clean.clean_ro_cf import clean_ro_cf
from dataprep.clean.clean_ro_cnp import clean_ro_cnp
from dataprep.clean.clean_ro_cui import clean_ro_cui
from dataprep.clean.clean_ro_onrc import clean_ro_onrc
from dataprep.clean.clean_isbn import clean_isbn

# with split
from dataprep.clean.clean_bic import clean_bic
from dataprep.clean.clean_bitcoin import clean_bitcoin
from dataprep.clean.clean_casrn import clean_casrn
from dataprep.clean.clean_cusip import clean_cusip
from dataprep.clean.clean_ean import clean_ean
from dataprep.clean.clean_figi import clean_figi
from dataprep.clean.clean_grid import clean_grid
from dataprep.clean.clean_iban import clean_iban
from dataprep.clean.clean_imei import clean_imei
from dataprep.clean.clean_imo import clean_imo
from dataprep.clean.clean_imsi import clean_imsi
from dataprep.clean.clean_isan import clean_isan
from dataprep.clean.clean_isil import clean_isil
from dataprep.clean.clean_isin import clean_isin
from dataprep.clean.clean_ismn import clean_ismn
from dataprep.clean.clean_issn import clean_issn
from dataprep.clean.clean_ad_nrt import clean_ad_nrt
from dataprep.clean.clean_al_nipt import clean_al_nipt
from dataprep.clean.clean_ar_cbu import clean_ar_cbu
from dataprep.clean.clean_ar_cuit import clean_ar_cuit
from dataprep.clean.clean_ar_dni import clean_ar_dni
from dataprep.clean.clean_at_uid import clean_at_uid
from dataprep.clean.clean_at_vnr import clean_at_vnr
from dataprep.clean.clean_lei import clean_lei
from dataprep.clean.clean_meid import clean_meid
from dataprep.clean.clean_vatin import clean_vatin

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


clean_list = [
    "clean_email",
    "clean_country",
    "clean_ip",
    "clean_phone",
    "clean_url",
    "clean_headers",
    "clean_date",
    "clean_lat_long",
    "clean_json",
    "clean_address",
    "clean_df",
    "clean_duplication",
    "clean_currency",
    "clean_au_abn",
    "clean_au_acn",
    "clean_au_tfn",
    "clean_be_iban",
    "clean_be_vat",
    "clean_bg_egn",
    "clean_bg_pnf",
    "clean_bg_vat",
    "clean_br_cnpj",
    "clean_br_cpf",
    "clean_by_unp",
    "clean_ca_bn",
    "clean_ca_sin",
    "clean_ch_esr",
    "clean_ch_ssn",
    "clean_ch_uid",
    "clean_ch_vat",
    "clean_cl_rut",
    "clean_cn_ric",
    "clean_cn_uscc",
    "clean_co_nit",
    "clean_cr_cpf",
    "clean_cr_cpj",
    "clean_cr_cr",
    "clean_cu_ni",
    "clean_cy_vat",
    "clean_cz_dic",
    "clean_cz_rc",
    "clean_de_handelsregisternummer",
    "clean_de_idnr",
    "clean_de_stnr",
    "clean_de_vat",
    "clean_de_wkn",
    "clean_dk_cpr",
    "clean_dk_cvr",
    "clean_do_cedula",
    "clean_do_ncf",
    "clean_do_rnc",
    "clean_ec_ci",
    "clean_ec_ruc",
    "clean_ee_ik",
    "clean_ee_kmkr",
    "clean_ee_registrikood",
    "clean_es_ccc",
    "clean_es_cif",
    "clean_es_cups",
    "clean_es_dni",
    "clean_es_iban",
    "clean_es_nie",
    "clean_es_nif",
    "clean_es_referenciacatastral",
    "clean_eu_at_02",
    "clean_eu_banknote",
    "clean_eu_eic",
    "clean_eu_nace",
    "clean_eu_vat",
    "clean_fi_alv",
    "clean_fi_associationid",
    "clean_fi_hetu",
    "clean_fi_veronumero",
    "clean_fi_ytunnus",
    "clean_fr_nif",
    "clean_fr_nir",
    "clean_fr_siren",
    "clean_fr_siret",
    "clean_fr_tva",
    "clean_gb_nhs",
    "clean_gb_sedol",
    "clean_gb_upn",
    "clean_gb_utr",
    "clean_gb_vat",
    "clean_gr_amka",
    "clean_gr_vat",
    "clean_gt_nit",
    "clean_hr_oib",
    "clean_hu_anum",
    "clean_id_npwp",
    "clean_ie_pps",
    "clean_ie_vat",
    "clean_il_hp",
    "clean_il_idnr",
    "clean_in_aadhaar",
    "clean_in_pan",
    "clean_is_kennitala",
    "clean_is_vsk",
    "clean_it_aic",
    "clean_it_codicefiscale",
    "clean_it_iva",
    "clean_jp_cn",
    "clean_kr_brn",
    "clean_kr_rrn",
    "clean_li_peid",
    "clean_lt_pvm",
    "clean_lt_asmens",
    "clean_lu_tva",
    "clean_lv_pvn",
    "clean_mc_tva",
    "clean_md_idno",
    "clean_me_iban",
    "clean_mt_vat",
    "clean_mu_nid",
    "clean_mx_curp",
    "clean_mx_rfc",
    "clean_my_nric",
    "clean_nl_brin",
    "clean_nl_btw",
    "clean_nl_bsn",
    "clean_nl_onderwijsnummer",
    "clean_nl_postcode",
    "clean_no_fodselsnummer",
    "clean_no_iban",
    "clean_no_kontonr",
    "clean_no_mva",
    "clean_no_orgnr",
    "clean_nz_bankaccount",
    "clean_nz_ird",
    "clean_pe_cui",
    "clean_pe_ruc",
    "clean_pl_nip",
    "clean_pl_pesel",
    "clean_pl_regon",
    "clean_pt_nif",
    "clean_py_ruc",
    "clean_ro_cf",
    "clean_ro_cnp",
    "clean_ro_cui",
    "clean_ro_onrc",
    "clean_isbn",
    "clean_bic",
    "clean_bitcoin",
    "clean_casrn",
    "clean_cusip",
    "clean_ean",
    "clean_figi",
    "clean_grid",
    "clean_iban",
    "clean_imei",
    "clean_imo",
    "clean_imsi",
    "clean_isan",
    "clean_isil",
    "clean_isin",
    "clean_ismn",
    "clean_issn",
    "clean_ad_nrt",
    "clean_al_nipt",
    "clean_ar_cbu",
    "clean_ar_cuit",
    "clean_ar_dni",
    "clean_at_uid",
    "clean_at_vnr",
    "clean_lei",
    "clean_meid",
    "clean_vatin",
]
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
    "clean_json": clean_json,
    "clean_address": clean_address,
    "clean_df": clean_df,
    # "clean_duplication": clean_duplication,
    "clean_currency": clean_currency,
    "clean_au_abn": clean_au_abn,
    "clean_au_acn": clean_au_acn,
    "clean_au_tfn": clean_au_tfn,
    "clean_be_iban": clean_be_iban,
    "clean_be_vat": clean_be_vat,
    "clean_bg_egn": clean_bg_egn,
    "clean_bg_pnf": clean_bg_pnf,
    "clean_bg_vat": clean_bg_vat,
    "clean_br_cnpj": clean_br_cnpj,
    "clean_br_cpf": clean_br_cpf,
    "clean_by_unp": clean_by_unp,
    "clean_ca_bn": clean_ca_bn,
    "clean_ca_sin": clean_ca_sin,
    "clean_ch_esr": clean_ch_esr,
    "clean_ch_ssn": clean_ch_ssn,
    "clean_ch_uid": clean_ch_uid,
    "clean_ch_vat": clean_ch_vat,
    "clean_cl_rut": clean_cl_rut,
    "clean_cn_ric": clean_cn_ric,
    "clean_cn_uscc": clean_cn_uscc,
    "clean_co_nit": clean_co_nit,
    "clean_cr_cpf": clean_cr_cpf,
    "clean_cr_cpj": clean_cr_cpj,
    "clean_cr_cr": clean_cr_cr,
    "clean_cu_ni": clean_cu_ni,
    "clean_cy_vat": clean_cy_vat,
    "clean_cz_dic": clean_cz_dic,
    "clean_cz_rc": clean_cz_rc,
    "clean_de_handelsregisternummer": clean_de_handelsregisternummer,
    "clean_de_idnr": clean_de_idnr,
    "clean_de_stnr": clean_de_stnr,
    "clean_de_vat": clean_de_vat,
    "clean_de_wkn": clean_de_wkn,
    "clean_dk_cpr": clean_dk_cpr,
    "clean_dk_cvr": clean_dk_cvr,
    "clean_do_cedula": clean_do_cedula,
    "clean_do_ncf": clean_do_ncf,
    "clean_do_rnc": clean_do_rnc,
    "clean_ec_ci": clean_ec_ci,
    "clean_ec_ruc": clean_ec_ruc,
    "clean_ee_ik": clean_ee_ik,
    "clean_ee_kmkr": clean_ee_kmkr,
    "clean_ee_registrikood": clean_ee_registrikood,
    "clean_es_ccc": clean_es_ccc,
    "clean_es_cif": clean_es_cif,
    "clean_es_cups": clean_es_cups,
    "clean_es_dni": clean_es_dni,
    "clean_es_iban": clean_es_iban,
    "clean_es_nie": clean_es_nie,
    "clean_es_nif": clean_es_nif,
    "clean_es_referenciacatastral": clean_es_referenciacatastral,
    "clean_eu_at_02": clean_eu_at_02,
    "clean_eu_banknote": clean_eu_banknote,
    "clean_eu_eic": clean_eu_eic,
    "clean_eu_nace": clean_eu_nace,
    "clean_eu_vat": clean_eu_vat,
    "clean_fi_alv": clean_fi_alv,
    "clean_fi_associationid": clean_fi_associationid,
    "clean_fi_hetu": clean_fi_hetu,
    "clean_fi_veronumero": clean_fi_veronumero,
    "clean_fi_ytunnus": clean_fi_ytunnus,
    "clean_fr_nif": clean_fr_nif,
    "clean_fr_nir": clean_fr_nir,
    "clean_fr_siren": clean_fr_siren,
    "clean_fr_siret": clean_fr_siret,
    "clean_fr_tva": clean_fr_tva,
    "clean_gb_nhs": clean_gb_nhs,
    "clean_gb_sedol": clean_gb_sedol,
    "clean_gb_upn": clean_gb_upn,
    "clean_gb_utr": clean_gb_utr,
    "clean_gb_vat": clean_gb_vat,
    "clean_gr_amka": clean_gr_amka,
    "clean_gr_vat": clean_gr_vat,
    "clean_gt_nit": clean_gt_nit,
    "clean_hr_oib": clean_hr_oib,
    "clean_hu_anum": clean_hu_anum,
    "clean_id_npwp": clean_id_npwp,
    "clean_ie_pps": clean_ie_pps,
    "clean_ie_vat": clean_ie_vat,
    "clean_il_hp": clean_il_hp,
    "clean_il_idnr": clean_il_idnr,
    "clean_in_aadhaar": clean_in_aadhaar,
    "clean_in_pan": clean_in_pan,
    "clean_is_kennitala": clean_is_kennitala,
    "clean_is_vsk": clean_is_vsk,
    "clean_it_aic": clean_it_aic,
    "clean_it_codicefiscale": clean_it_codicefiscale,
    "clean_it_iva": clean_it_iva,
    "clean_jp_cn": clean_jp_cn,
    "clean_kr_brn": clean_kr_brn,
    "clean_kr_rrn": clean_kr_rrn,
    "clean_li_peid": clean_li_peid,
    "clean_lt_pvm": clean_lt_pvm,
    "clean_lt_asmens": clean_lt_asmens,
    "clean_lu_tva": clean_lu_tva,
    "clean_lv_pvn": clean_lv_pvn,
    "clean_mc_tva": clean_mc_tva,
    "clean_md_idno": clean_md_idno,
    "clean_me_iban": clean_me_iban,
    "clean_mt_vat": clean_mt_vat,
    "clean_mu_nid": clean_mu_nid,
    "clean_mx_curp": clean_mx_curp,
    "clean_mx_rfc": clean_mx_rfc,
    "clean_my_nric": clean_my_nric,
    "clean_nl_brin": clean_nl_brin,
    "clean_nl_btw": clean_nl_btw,
    "clean_nl_bsn": clean_nl_bsn,
    "clean_nl_onderwijsnummer": clean_nl_onderwijsnummer,
    "clean_nl_postcode": clean_nl_postcode,
    "clean_no_fodselsnummer": clean_no_fodselsnummer,
    "clean_no_iban": clean_no_iban,
    "clean_no_kontonr": clean_no_kontonr,
    "clean_no_mva": clean_no_mva,
    "clean_no_orgnr": clean_no_orgnr,
    "clean_nz_bankaccount": clean_nz_bankaccount,
    "clean_nz_ird": clean_nz_ird,
    "clean_pe_cui": clean_pe_cui,
    "clean_pe_ruc": clean_pe_ruc,
    "clean_pl_nip": clean_pl_nip,
    "clean_pl_pesel": clean_pl_pesel,
    "clean_pl_regon": clean_pl_regon,
    "clean_pt_nif": clean_pt_nif,
    "clean_py_ruc": clean_py_ruc,
    "clean_ro_cf": clean_ro_cf,
    "clean_ro_cnp": clean_ro_cnp,
    "clean_ro_cui": clean_ro_cui,
    "clean_ro_onrc": clean_ro_onrc,
    "clean_isbn": clean_isbn,
    "clean_bic": clean_bic,
    "clean_bitcoin": clean_bitcoin,
    "clean_casrn": clean_casrn,
    "clean_cusip": clean_cusip,
    "clean_ean": clean_ean,
    "clean_figi": clean_figi,
    "clean_grid": clean_grid,
    "clean_iban": clean_iban,
    "clean_imei": clean_imei,
    "clean_imo": clean_imo,
    "clean_imsi": clean_imsi,
    "clean_isan": clean_isan,
    "clean_isil": clean_isil,
    "clean_isin": clean_isin,
    "clean_ismn": clean_ismn,
    "clean_issn": clean_issn,
    "clean_ad_nrt": clean_ad_nrt,
    "clean_al_nipt": clean_al_nipt,
    "clean_ar_cbu": clean_ar_cbu,
    "clean_ar_cuit": clean_ar_cuit,
    "clean_ar_dni": clean_ar_dni,
    "clean_at_uid": clean_at_uid,
    "clean_at_vnr": clean_at_vnr,
    "clean_lei": clean_lei,
    "clean_meid": clean_meid,
    "clean_vatin": clean_vatin,
}

contain_birthday_place = ["clean_cn_ric", "clean_my_nric"]
contain_isin = ["clean_de_wkn", "clean_gb_sedol", "clean_cusip"]
contain_mask = ["clean_in_aadhaar", "clean_in_pan"]
contain_iban = ["clean_no_kontonr", "clean_es_ccc"]
contain_gender = [
    "clean_cu_ni",
    "clean_ee_ik",
    "clean_gr_amka",
    "clean_mx_curp",
    "clean_it_codicefiscale",
    "clean_no_fodselsnummer",
    "clean_pl_pesel",
]
contain_birthdate = [
    "clean_bg_egn",
    "clean_dk_cpr",
    "clean_kr_rrn",
    "clean_mu_nid",
    "clean_lt_asmens",
    "clean_lv_pvn",
    "clean_ro_cnp",
]
contain_split = [
    "clean_bic",
    "clean_bitcoin",
    "clean_casrn",
    "clean_cusip",
    "clean_ean",
    "clean_figi",
    "clean_grid",
    "clean_iban",
    "clean_imei",
    "clean_imo",
    "clean_imsi",
    "clean_isan",
    "clean_isil",
    "clean_isin",
    "clean_ismn",
    "clean_issn",
    "clean_ad_nrt",
    "clean_al_nipt",
    "clean_ar_cbu",
    "clean_ar_cuit",
    "clean_ar_dni",
    "clean_at_uid",
    "clean_at_vnr",
    "clean_lei",
    "clean_meid",
    "clean_vatin",
]
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

    args = inspect.signature(clean_function_dic[clean_func]).parameters
    args_list = list(args.keys())

    for arg in args_list:
        temp_option_list = []
        if arg in ("df", "column"):
            continue
        if arg == "inplace":
            break

        if clean_func == "clean_email":
            if isinstance(args[arg].default, bool):
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_headers":
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
            if arg in ["remove_auth", "split"]:
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_address":
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
        # elif clean_func == "clean_duplication":
        #     if arg == "page_size":
        #         for i in range(5, 8):
        #             temp_option_list.append({"value": str(i), "label": str(i)})
        #         param_dic[arg] = temp_option_list
        #         param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_currency":
            if arg == "input_currency":
                temp_option_list.append({"value": "usd", "label": "USD"})
                temp_option_list.append(
                    {
                        "value": "Binance Coin",
                        "label": "Binance Coin",
                    }
                )
                temp_option_list.append(
                    {
                        "value": "ethereum",
                        "label": "Ethereum",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "target_representation":
                temp_option_list.append(
                    {"value": "decimal", "label": "Decimal(floating point number)"}
                )
                temp_option_list.append(
                    {
                        "value": "abbreviation",
                        "label": "Abbreviation((string with comma seprated values)",
                    }
                )
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_au_acn":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "abn", "label": "Australian Business Number"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_birthday_place:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})
                temp_option_list.append({"value": "birthplace", "label": "Birthplace"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_gender:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})
                temp_option_list.append({"value": "gender", "label": "Gender"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_isin:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "isin", "label": "ISIN"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_birthdate:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "birthdate", "label": "Birthdate"})

            param_dic[arg] = temp_option_list
            param_default[arg] = str(args[arg].default)
        elif clean_func in contain_iban:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "iban", "label": "IBAN"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_es_iban":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "ccc", "label": "Código Cuenta Corriente"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_eu_nace":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "label", "label": "Label"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_eu_vat":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "country", "label": "Country"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_fr_siren":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "tva", "label": "TVA"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func in contain_mask:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "mask", "label": "Mask"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_it_aic":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "base10", "label": "BASE10"})
                temp_option_list.append({"value": "base32", "label": "BASE32"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        elif clean_func == "clean_no_iban":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append(
                    {"value": "kontonr", "label": "Norwegian bank account part"}
                )

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_nz_bankaccount":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "info", "label": "Supplied number"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_pe_cui":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "ruc", "label": "RUC"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_pe_ruc":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "dni", "label": "DNI (CUI)"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func == "clean_isbn":
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                temp_option_list.append({"value": "base10", "label": "BASE10"})
                temp_option_list.append({"value": "base13", "label": "BASE13"})

                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
        elif clean_func in contain_split:
            if arg == "output_format":
                if clean_func in contain_isin:
                    temp_option_list.append({"value": "isin", "label": "ISIN"})
                if clean_func == "clean_isan":
                    temp_option_list.append({"value": "binary", "label": "Binary"})
                    temp_option_list.append({"value": "urn", "label": "URN"})
                    temp_option_list.append({"value": "xml", "label": "XML"})
                if clean_func == "clean_issn":
                    temp_option_list.append({"value": "ean", "label": "EAN"})
                if clean_func == "clean_meid":
                    temp_option_list.append({"value": "hex", "label": "Hex"})
                    temp_option_list.append({"value": "dec", "label": "Decimal"})
                    temp_option_list.append({"value": "binary", "label": "Binary"})
                    temp_option_list.append({"value": "pseudo_esn", "label": "Pseudo_esn"})

                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)
            if arg == "split":
                temp_option_list.append({"value": "True", "label": "True"})
                temp_option_list.append({"value": "False", "label": "False"})
                param_dic[arg] = temp_option_list
                param_default[arg] = str(args[arg].default)

        else:
            if arg == "output_format":
                temp_option_list.append({"value": "standard", "label": "Standard"})
                temp_option_list.append({"value": "compact", "label": "Compact"})
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
            if clean_func in clean_list:
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
            "clean_json",
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
        else:
            df_cleaned = clean_function_dic[clean_func](
                index_df, column=selected_col, **selected_params
            )
        print(df_cleaned)
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
