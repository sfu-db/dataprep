<template>
  <el-container>
    <el-header>
      <Header
        @newDataYes="updateTableData"
        @updateFooterLog="reloadFooterLog"
        @getTheTrueFile="updateTableData"/>
    </el-header>
    <el-main>
      <el-row class="el-main-params">
        <el-col :span="4">  
         <Recipe
            @optionsValue="changeCleanning"
            @parashow="paraShowChange"/>
        </el-col>  
        <el-col :span="10" class="block-div">  
          <MainTable :key="timer2" v-bind:filenname="filenname"/>
        </el-col>  
        <el-col :span="4">  
        <Issue
          @optionsValue="changeCleanning"
          @parashow="paraShowChange"/>
        </el-col> 
        <Suggestion
          @optionsValue="changeCleanning"
          @parashow="paraShowChange"/>
       
      </el-row>
    </el-main>
  </el-container>
</template>

<script>
import Header from "./Header";
import Recipe from "./Recipe";
import MainTable from "./MainTable";
import FooterLog from "./FooterLog";
import CleanWholeDF from "./CleanWholeDF";
import CleanFunctions from "./CleanFunctions";
import Suggestion from "./Suggestion"
import Issue from "./Issue"

export default {
  name: "SampleDF",
  components: {
    Header,
    Recipe,
    MainTable,
    FooterLog,
    CleanWholeDF,
    CleanFunctions,    
    Suggestion,
    Issue
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
  border: 3px solid #000;
}

.el-main {
  color: #000;
  text-align: center;
  padding: 0;
  margin: 0;
}

.el-main-params {
  margin-top: 10px;
  /* height: 500px; */
  border: 3px solid #000;
}

.el-main-1st-row {
  margin-top: 5px;
  border: 3px solid #000;
}

.el-footer {
  color: #333;
  text-align: center;
  height: 100px !important;
  padding: 0;
  margin-top: 5px;
  border: 1px solid #000;
}
.el-main-1st-row-2ndcol {
  border-left: 2px solid #000;
  height: 100%;
}

.block-div {
  border: groove;
}
</style>
