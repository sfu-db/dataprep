<template>
  <el-row class="el-main-1strow">
    <el-col :span="10" class="el-main-1strow-1stcol">
      <img src="../assets/cleanning.png" class="cleanning_img" /><br />Clean
    </el-col>
    <el-col :span="14">
      <el-button @click="clickDataFrame">Whole DF</el-button><br />
      <el-select v-model="value" placeholder="Single Col">
        <el-option
          v-for="(value, label) in options"
          :key="label"
          :label="value"
          :value="label"
        >
        </el-option>
      </el-select>
    </el-col>
  </el-row>
</template>

<script>
import axios from "axios";
import endpoint from "../util/endpoint";

export default {
  name: "Cleanning",
  data() {
    return {
      options: {
        clean_email: "Email",
        clean_headers: "Header",
        clean_country: "Country",
        clean_date: "Date",
        clean_lat_long: "Coordinate",
        clean_ip: "IP address",
        clean_phone: "Phone Number",
        clean_json: "JSON",
        // clean_text: "Text",
        clean_url: "URL",
        clean_address: "Address",
        // clean_duplication: "Duplication",
        clean_currency: "Currency",
        clean_au_abn: "Australian Business Numbers",
        clean_au_acn: "Australian Company Numbers",
        clean_au_tfn: "Australian Tax File Numbers",
        clean_be_iban: "Belgian IBAN Numbers",
        clean_be_vat: "Belgian VAT Numbers",
        clean_bg_egn: "Bulgarian National Identification Numbers",
        clean_bg_pnf: "Belarusian UNP Numbers",
        clean_bg_vat: "Bulgarian VAT Numbers",
        clean_br_cnpj: "Brazilian Company Identifiers",
        clean_br_cpf: "Brazilian National Identifiers",
        clean_by_unp: "Belarusian UNP Numbers",
        clean_ca_bn: "Canadian Business Numbers",
        clean_ca_sin: "Canadian Social Insurance Numbers",
        clean_ch_esr: "Swiss Einzahlungsschein MIT Referenznummers",
        clean_ch_ssn: "Swiss Social Security Numbers",
        clean_ch_uid: "Swiss Business Identifiers",
        clean_ch_vat: "Swiss VAT Numbers",
        clean_cl_rut: "Chile RUT/RUN Numbers",
        clean_cn_ric: "Chinese Resident Identity Card Numbers",
        clean_cn_uscc: "Chinese Unified Social Credit Codes",
        clean_co_nit: "Colombian Identity Codes",
        clean_cr_cpf: "Costa Rica Physical Person ID Numbers",
        clean_cr_cpj: "Costa Rica Tax Numbers",
        clean_cr_cr: "Costa Rica Foreigners ID Numbers",
        clean_cu_ni: "Cuban Identity Card Numbers",
        clean_cy_vat: "Cypriot VAT Numbers",
        clean_cz_dic: "Czech VAT Numbers",
        clean_cz_rc: "Czech Birth Numbers",
        clean_de_handelsregisternummer: "German Company Registry IDs",
        clean_de_idnr: "German Personal Tax Numbers",
        clean_de_stnr: "German Tax Numbers",
        clean_de_vat: "German VAT Numbers",
        clean_de_wkn: "German Securities Identification Codes",
        clean_dk_cpr: "Danish Citizen Numbers",
        clean_dk_cvr: "Danish CVR Numbers",
        clean_do_cedula: "Dominican Republic National Identifiers",
        clean_do_ncf: "Dominican Republic Invoice Numbers",
        clean_do_rnc: "Dominican Republic Tax Registrations",
        clean_ec_ci: "Ecuadorian Personal Identity Codes",
        clean_ec_ruc: "Ecuadorian Company Tax Numbers",
        clean_ee_ik: "stonian Personcal ID Numbers",
        clean_ee_kmkr: "Estonian KMKR Numbers",
        clean_ee_registrikood: "Estonian Organisation Registration Codes",
        clean_es_ccc: "Spanish Bank Account Codes",
        clean_es_cif: "Spanish Fiscal Numbers",
        clean_es_cups: "Spanish Meter Point Numbers",
        clean_es_dni: "Spanish Personal Identity Codes",
        clean_es_iban: "Spanish IBANs",
        clean_es_nie: "Spanish Foreigner Identity Codes",
        clean_es_nif: "Spanish NIF Numbers",
        clean_es_referenciacatastral: "Spanish Real State IDs",
        clean_eu_at_02: "SEPA Identifier of the Creditor",
        clean_eu_banknote: "Euro Banknote Serial Numbers",
        clean_eu_eic: "European Energy Identification Codes",
        clean_eu_nace: "Classification For Businesses In The European Union",
        clean_eu_vat: "European VAT Numbers",
        clean_fi_alv: "Finnish ALV Numbers",
        clean_fi_associationid: "Finnish Association Registry IDs",
        clean_fi_hetu: "Finnish Personal Identity Codes",
        clean_fi_veronumero: "Finnish Individual Tax Numbers",
        clean_fi_ytunnus: "Finnish Business Identifiers",
        clean_fr_nif: "French Tax Identification Numbers",
        clean_fr_nir: "French Personal Identification Numbers",
        clean_fr_siren: "French Company Identification Numbers",
        clean_fr_siret: "French Company Establishment Identification Numbers",
        clean_fr_tva: "French TVA Numbers",
        clean_gb_nhs:
          "United Kingdom National Health Service Patient Identifiers",
        clean_gb_sedol: "Stock Exchange Daily Official List Numbers",
        clean_gb_upn: "English Unique Pupil Numbers",
        clean_gb_utr: "United Kingdom Unique Taxpayer References",
        clean_gb_vat: "United Kingdom VAT Numbers",
        clean_gr_amka: "Greek Social Security Numbers",
        clean_gr_vat: "Greek VAT Numbers",
        clean_gt_nit: "Guatemala Tax Numbers",
        clean_hr_oib: "Croatian Identification Numbers",
        clean_hu_anum: "Hungarian ANUM Numbers",

        clean_id_npwp: "Indonesian VAT Numbers",
        clean_ie_pps: "Irish Personal Numbers",
        clean_ie_vat: "Irish VAT Numbers",
        clean_il_hp: "Israeli Company Numbers",
        clean_il_idnr: "Israeli Personal Numbers",
        clean_in_aadhaar: "Indian Digital Resident Personal Identity Numbers",
        clean_in_pan: "Indian Permanent Account Numbers",
        clean_is_kennitala: "Icelandic Identity Codes",
        clean_is_vsk: "Icelandic VSK Numbers",
        clean_it_aic: "Italian Code For Identification Of Drugs",
        clean_it_codicefiscale: "Italian Fiscal Codes",
        clean_it_iva: "Italian IVA Numbers",
        clean_jp_cn: "Japanese Corporate Numbers",
        clean_kr_brn: "South Korea Business Registration Numbers",
        clean_kr_rrn: "South Korean Resident Registration Numbers",
        clean_li_peid: "Liechtenstein Tax Code For Individuals And Entities",
        clean_lt_pvm: "Lithuanian PVM Numbers",
        clean_lt_asmens: "Lithuanian Personal Numbers",
        clean_lu_tva: "Luxembourgian TVA Numbers",
        clean_lv_pvn: "Latvian PVN (VAT) Numbers",
        clean_mc_tva: "Monacan TVA Numbers",
        clean_md_idno: "Moldavian Company Identification Numbers",
        clean_me_iban: "Montenegro IBANs",
        clean_mt_vat: "Maltese VAT Numbers",
        clean_mu_nid: "Mauritian National ID Numbers",
        clean_mx_curp: "Mexican Personal Identifiers",
        clean_mx_rfc: "Mexican Tax Numbers",
        clean_my_nric: "Malaysian National Registration Identity Card Numbers",
        clean_nl_brin: "BRIN Numbers",
        clean_nl_btw: "Dutch BTW Numbers",
        clean_nl_bsn: "Dutch Citizen Identification Numbers",
        clean_nl_onderwijsnummer: "Dutch Student Identification Numbers",
        clean_nl_postcode: "Dutch Postal Codes",
        clean_no_fodselsnummer: "Norwegian Birth Numbers",
        clean_no_iban: "Norwegian IBANs",
        clean_no_kontonr: "Norwegian Bank Account Numbers",
        clean_no_mva: "Norwegian VAT Numbers",
        clean_no_orgnr: "Norwegian Organisation Numbers",
        clean_nz_bankaccount: "New Zealand Bank Account Numbers",
        clean_nz_ird: "New Zealand IRD Numbers",
        clean_pe_cui: "Peruvian Personal Numbers",
        clean_pe_ruc: "Peruvian Fiscal Numbers",
        clean_pl_nip: "Polish VAT Numbers",
        clean_pl_pesel: "Polish National Identification Numbers",
        clean_pl_regon: "Polish Register Of Economic Units",
        clean_pt_nif: "Portuguese NIF Numbers",
        clean_py_ruc: "Paraguay RUC Numbers",
        clean_ro_cf: "Romanian CF (VAT) Numbers",
        clean_ro_cnp: "Romanian Numerical Personal Codes",
        clean_ro_cui: "Romanian Company Identifiers",
        clean_ro_onrc: "Romanian Trade Register Identifiers",
        clean_isbn: "ISBN Numbers",
        clean_bic: "ISO 9362 Business identifier codes",
        clean_bitcoin: "Bitcoin Addresses",
        clean_casrn: "CAS Registry Numbers",
        clean_cusip: "CUSIP numbers",
        clean_ean: "EAN (International Article Number)",
        clean_figi: "Financial Instrument Global Identifier (FIGI) Numbers",
        clean_grid: "Global Release Identifier (GRID) numbers",
        clean_iban: "IBAN numbers",
        clean_imei: "International Mobile Equipment Identity (IMEI) numbers",
        clean_imo: "International Maritime Organization Numbers",
        clean_imsi: "International Mobile Subscriber Identity (IMSI) numbers",
        clean_isan: "International Standard Audiovisual Numbers",
        clean_isil:
          "International Standard Identifier for Libraries (ISIL) numbers",
        clean_isin:
          "International Securities Identification Number (ISIN) numbers",
        clean_ismn: "ISMN number",
        clean_issn: "International Standard Serial Numbers",
        clean_ad_nrt:
          "Andorra NRT (Número de Registre Tributari, Andorra tax number)",
        clean_al_nipt:
          "NIPT (Numri i Identifikimit për Personin e Tatueshëm, Albanian VAT number)",
        clean_ar_cbu:
          "CBU (Clave Bancaria Uniforme, Argentine bank account number)",
        clean_ar_cuit:
          "CUIT (Código Único de Identificación Tributaria, Argentinian tax number)",
        clean_ar_dni:
          "DNI (Documento Nacional de Identidad, Argentinian national identity nr.)",
        clean_at_uid:
          "UID (Umsatzsteuer-Identifikationsnummer, Austrian VAT number)",
        clean_at_vnr:
          "VNR, SVNR, VSNR (Versicherungsnummer, Austrian social security number)",
        clean_lei: "Legal Entity Identifier (LEI) Numbers",
        clean_meid: "Mobile Equipment Identifiers (MEIDs)",
        clean_vatin: "International value added tax identification numbers",
      },
      value: "",
      isWholeDF: false,
      parashow: "",
    };
  },
  methods: {
    clickDataFrame: function () {
      this.isWholeDF = true;
      this.parashow = new Date().getTime();
      this.value = "";
    },
  },
  watch: {
    value: function (nval, oval) {
      let _this = this;
      if (!(nval == "" && _this.isWholeDF)) {
        _this.isWholeDF = false;
        axios
          .post(endpoint.API_URL + "getFunctionParams", {
            clean_func: nval,
          })
          .then(function (res) {
            let tableColumns = res.data.tableColumns;
            let paramDic = res.data.paramDic;
            let paramDefault = res.data.paramDefault;

            _this.$emit(
              "optionsValue",
              nval,
              tableColumns,
              paramDic,
              paramDefault
            );
          });
      }
    },
    parashow: function (nval, oval) {
      this.$emit("parashow", nval);
    },
  },
};
</script>

<style scoped>
.cleanning_img {
  display: inline;
  width: 45px;
  height: 40px;
  margin-top: 20px;
  margin-bottom: 10px;
}
.el-button {
  margin-top: 6px;
  width: 150px;
}
.el-select {
  margin-top: 6px;
  width: 150px;
  margin-bottom: 40px;
}
</style>
