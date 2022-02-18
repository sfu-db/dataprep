<template>
  <el-row class="el-main-1strow">
    <el-col :span="10" class="el-main-1strow-1stcol">
      <img src="../assets/cleanning.png" class="cleanning_img" /><br />Clean
    </el-col>
    <el-col :span="14">
      <el-button @click="clickDataFrame">Whole DF</el-button><br />
      <el-select v-model="value" placeholder="Single Col">
        <el-option
          v-for="item in options"
          :key="item.value"
          :label="item.label"
          :value="item.value"
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
      options: [
        {
          value: "clean_email",
          label: "Email",
        },
        {
          value: "clean_headers",
          label: "Headers",
        },
        {
          value: "clean_country",
          label: "Country",
        },
        {
          value: "clean_date",
          label: "Date",
        },
        {
          value: "clean_lat_long",
          label: "Coordinate",
        },
        {
          value: "clean_ip",
          label: "IP address",
        },
        {
          value: "clean_phone",
          label: "Phone Number",
        },
        //{
        //  value: "clean_text",
        //  label: "Text",
        //},
        {
          value: "clean_url",
          label: "URL",
        },
        {
          value: "clean_address",
          label: "Address",
        },
      ],
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
            //_this.options0 = res.data.tableColumns
            //_this.list = res.data.operationLog
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
