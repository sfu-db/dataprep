<template>
  <el-row>
    <el-col :span="4">
      <img src="../assets/para.png" class="para-img" /><br />Params
    </el-col>
    <el-col :span="3" class="el-col-select-box">
      clean_headers<br /><br />
      <el-select v-model="cleanHeaderValue">
        <el-option
          v-for="item in cleanHeaderOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        >
        </el-option>
      </el-select>
    </el-col>
    <el-col :span="3" :offset="1" class="el-col-select-box">
      data_type_detection<br /><br />
      <el-select v-model="dataTypeDetectionValue">
        <el-option
          v-for="item in dataTypeDetectionOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        >
        </el-option>
      </el-select>
    </el-col>
    <el-col :span="3" :offset="2" class="el-col-select-box">
      standardize_missing<br /><br />
      <el-select v-model="standardizeMissingValue">
        <el-option
          v-for="item in standardizeMissingOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        >
        </el-option>
      </el-select>
    </el-col>
    <el-col :span="3" :offset="2" class="el-col-select-box">
      downcast_memory<br /><br />
      <el-select v-model="downcastMemoValue">
        <el-option
          v-for="item in downcastMemoOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
        >
        </el-option>
      </el-select>
    </el-col>
    <el-col :span="3">
      <el-button
        icon="el-icon-circle-check"
        class="el-col-select-box-button"
        @click="clickOK"
        ><br />OK!</el-button
      >
    </el-col>
  </el-row>
</template>
<script>
import axios from "axios";
import endpoint from "../util/endpoint";

export default {
  name: "CleanWholeDF",
  data() {
    return {
      cleanHeaderOptions: [
        {
          value: "True",
          label: "True",
        },
      ],
      cleanHeaderValue: "True",

      dataTypeDetectionOptions: [
        {
          value: "semantic",
          label: "semantic",
        },
        {
          value: "atomic",
          label: "atomic",
        },
        {
          value: "none",
          label: "none",
        },
      ],
      dataTypeDetectionValue: "semantic",

      standardizeMissingOptions: [
        {
          value: "remove",
          label: "remove",
        },
        {
          value: "fill",
          label: "fill",
        },
        {
          value: "ignore",
          label: "ignore",
        },
      ],
      standardizeMissingValue: "remove",

      downcastMemoOptions: [
        {
          value: "True",
          label: "True",
        },
        {
          value: "False",
          label: "False",
        },
      ],
      downcastMemoValue: "True",
    };
  },

  methods: {
    clickOK() {
      let _this = this;
      let param = new FormData();
      let config = {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      };

      param.append("clean_headers", _this.cleanHeaderValue);
      param.append("data_type_detection", _this.dataTypeDetectionValue);
      param.append("standardize_missing_values", _this.standardizeMissingValue);
      param.append("downcast_memory", _this.downcastMemoValue);
      param.append("new_log", "clean_df(df)");

      axios
        .post(endpoint.API_URL + "cleanWholeDF", param, config)
        .then(function (res) {
          _this.$emit("newDataYes", res.data);
          _this.$emit("updateFooterLog", "yes");
        })
        .catch(function (e) {
          console.log(e);
        });
    },
  },
};
</script>
<style scoped>
.para-img {
  width: 40px;
  height: 50px;
  margin-top: 10px;
  margin-left: 10px;
}
.el-col-select-box {
  padding-top: 10px;
  margin-left: 10px;
  width: 80px;
  word-break: break-all;
  word-wrap: break-word;
}
.el-col-select-box-button {
  margin-top: 30px;
  margin-left: 40px;
}
</style>
