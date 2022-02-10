<template>
  <el-container>
    <el-header
      ><Header
        @newDataYes="updateTableData"
        @updateFooterLog="reloadFooterLog"
        @getTheTrueFile="updateTableData"
    /></el-header>
    <el-main>
      <el-row class="el-main-params">
        <el-col :span="6"
          ><Cleanning
            @optionsValue="changeCleanning"
            @parashow="paraShowChange"
        /></el-col>
        <el-col :span="16" :offset="1" class="el-main-1st-row-2ndcol">
          <CleanWholeDF
            v-show="parashow"
            @updateFooterLog="reloadFooterLog"
            @newDataYes="updateTableData"
            v-bind:filenname="filenname"
          />
          <CleanFunctions
            v-show="singColoumShow"
            @clickButtonYes="reloadFooterLog"
            @newDataYes="updateTableData"
            @updateFooterLog="reloadFooterLog"
            v-bind:filenname="filenname"
            v-bind:clean_func="clean_func"
            v-bind:table_cols="table_cols"
            v-bind:param_dic="param_dic"
            v-bind:param_default="param_default"
          />
        </el-col>
      </el-row>
      <el-row class="el-main-1st-row">
        <el-col :span="24"
          ><MainTable :key="timer2" v-bind:filenname="filenname"
        /></el-col>
      </el-row>
    </el-main>
    <el-footer><FooterLog :key="timer" /></el-footer>
  </el-container>
</template>

<script>
import Header from "./Header";
import Cleanning from "./Cleanning";
import MainTable from "./MainTable";
import FooterLog from "./FooterLog";
import CleanWholeDF from "./CleanWholeDF";
import CleanFunctions from "./CleanFunctions";

export default {
  name: "SampleDF",
  components: {
    Header,
    Cleanning,
    MainTable,
    FooterLog,
    CleanWholeDF,
    CleanFunctions,
  },
  data() {
    return {
      clean_func: "",
      table_cols: [],
      param_dic: {},
      param_default: [],
      singColoumShow: false,
      parashow: false,
      timer: "",
      timer2: "",
      filenname: "yes",
    };
  },
  methods: {
    changeCleanning: function (nval, tableColumns, paramDic, paramDefault) {
      this.singColoumShow = true;
      this.parashow = false;
      this.clean_func = nval;
      this.table_cols = tableColumns;
      this.param_dic = paramDic;
      this.param_default = paramDefault;
    },
    paraShowChange: function (nval) {
      this.parashow = true;
      this.singColoumShow = false;
    },
    reloadFooterLog: function (para) {
      this.timer = new Date().getTime();
    },
    updateTableData: function (para) {
      this.timer2 = new Date().getTime();
      this.filenname = para;
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.el-container {
  min-height: 100%;
  padding: 0;
  margin: 0;
}
.el-header {
  color: #333;
  text-align: center;
  height: 10%;
  padding: 0;
  margin: 0;
  border: 3px solid #ccc;
}

.el-main {
  color: #333;
  text-align: center;
  padding: 0;
  margin: 0;
}

.el-main-params {
  margin-top: 5px;
  height: 130px;
  border: 3px solid #ccc;
}

.el-main-1st-row {
  margin-top: 5px;
  border: 3px solid #ccc;
}

.el-footer {
  color: #333;
  text-align: center;
  height: 100px !important;
  padding: 0;
  margin-top: 5px;
  border: 3px solid #ccc;
}
.el-main-1st-row-2ndcol {
  border-left: 2px solid #ccc;
  height: 100%;
}
</style>
