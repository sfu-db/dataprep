
<style scoped>
.header_center{
  display: flex;
  align-content: center;
  justify-content: center;
  /* border-style:solid;
  border-width:1px;  */
}
.shadow {
   margin: 10px;
   margin-bottom: 30px;
   box-shadow: 0 3px #999;
}
.el-row-col {
  line-height: 60px;
  font-size: 20px;
  font-weight: 600;
  margin: 10px;
}
.new-block {
  border-radius: 5px;
  background-color: rgb(209, 252, 194);
  margin-bottom: 20px;
  min-height: 100%;
}
.whole-block {
  border-radius: 5px;
  margin: 10px;
  min-height: 890px;
}
</style>


<template>
  <el-row >
    <div class="whole-block">
      <el-card class="shadow">
      <div class = "header_center">
        <b class = "el-row-col">Suggestions</b>
      </div>
        <div class="new-block ">
          <br/>  
          <b>Transform all email into</b>
          <br/>
          
          <el-select v-model="value" placeholder="Select">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>    
          </el-select>
          <el-button @click = "highlightCol" class="shadow"  type="success" >Preview</el-button>
          <el-button class="shadow"  type="primary" >Apply</el-button>
        </div>
        <div class="new-block ">
          <br/>  
          <b>Transform all header formula into</b>
          <br/>
          
          <el-select v-model="value" placeholder="Select">
            <el-option
              v-for="item in options"
              :key="item.value"
              :label="item.label"
              :value="item.value">
            </el-option>    
          </el-select>
          <el-button @click = "highlightCol" class="shadow"  type="success" >Preview</el-button>
          <el-button class="shadow"  type="primary" >Apply</el-button>
          <br/>
        </div>
    </el-card>
  </div>
  </el-row>
</template>

<script>
import axios from "axios";
import MainTable from "./MainTable.vue";
import endpoint from "../util/endpoint";

export default {
  name: "Suggestion",
  data() {
    return {
      value: "",
      isWholeDF: false,
      parashow: "",
      options: [{
          value: 'name_of_col',
          label: 'name_of_col'
        }, {
          value: 'name_of_row',
          label: 'name_of_row'
        }, {
          value: 'parameter_name',
          label: 'parameter_name'
        }],
    };
  },
  methods: {
    clickDataFrame: function () {
      this.isWholeDF = true;
      this.parashow = new Date().getTime();
      this.value = "";
    },
    highlightCol: function() {
      console.log(this.$root.$refs)
      MainTable.setCellColor();
    }
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
