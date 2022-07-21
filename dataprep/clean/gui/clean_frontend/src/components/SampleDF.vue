<template>
  <el-container>
    <!-- <el-col :span="23" >DataPrep.Clean UI</el-col> -->
    <el-header class="el-row-col">
      <img src="../assets/logo.png" class="img-size" />
      <!-- <Header
        @newDataYes="updateTableData"
        @updateFooterLog="reloadFooterLog"
        @getTheTrueFile="updateTableData"/> -->
    </el-header>
    <br/>

  <el-row :gutter="10" class="main-part">
    <el-col :span="6" >
      <div class="grid-content bg-purple-light">
        <Issue
        @optionsValue="changeCleanning"
        @parashow="paraShowChange"/>
        
        <Recipe
           @optionsValue="changeCleanning"
           @parashow="paraShowChange"/>
    </div>
  </el-col>
  <el-col :span="12">
    <div class="grid-content bg-purple-light">
        <MainTable :key="timer2" v-bind:filenname="filenname"/>
    </div>
  </el-col>
  <el-col :span="6" class="el-container">
  <div class="grid-content bg-purple-light">
      <Suggestion
          @optionsValue="changeCleanning"
          @parashow="paraShowChange"/>
  </div></el-col>
   </el-row>
      
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

.block-part {
  padding: 10;
  margin:10;
}

.main-part {
  background-color: #99a9bf;
  border-radius: 10px;
  min-height: 100%;
  padding: 15px;
  margin: 0;
}
.el-header {
  color: #333;
  text-align: center;
  height: 10%;
  padding: 0;
  margin: 0;
}

.el-main {
  color: #000;
  text-align: center;
  padding: 0;
  margin: 0;
}
.el-row-col {
  line-height: 60px;
  font-size: 25px;
  font-weight: 600;
  border-radius: 5px;
  background-color:  #D3DCE6;;
}
.el-main-params {
  margin-top: 10px;
}

.el-main-1st-row {
  margin-top: 5px;
  border: 3px solid #000;
}
.img-size {
  width: 200px;
  height: auto;
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
.el-col {
    border-radius: 4px;
  }
  .bg-purple-dark {
    background: #99a9bf;
  }
  .bg-purple {
    background: #d3dce6;
  }
  .bg-purple-light {
    background: #e5e9f2;
  }
  .grid-content {
    border-radius: 4px;
    min-height: 36px;
  }
</style>
