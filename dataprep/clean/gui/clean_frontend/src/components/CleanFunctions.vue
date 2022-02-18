<template>
  <el-row>
    <!--<el-col :span="2" class="para-img-col">
      <img src="../assets/columns.png" class="para-img" />
    </el-col>-->
    <el-form ref="rulesForm" :model="rulesForm" :rules="rules">
      <el-col :span="2" class="el-col-select-box0">
        <el-form-item
          label="Columns"
          :rules="[
            {
              required: true,
              message: 'Please select a column',
              trigger: 'blur',
            },
          ]"
        >
          <el-select
            v-model="initColValue"
            :rules="[{ required: true, trigger: 'blur' }]"
            placeholder="Select"
            @change="changeColSelection"
          >
            <el-option
              v-for="item in tableCols"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-form-item>
      </el-col>
    </el-form>
    <el-col :span="2" class="para-img-col">
      <img src="../assets/para.png" class="para-img" /><br />Params
    </el-col>
    <el-col
      :span="2"
      class="el-col-select-box"
      v-for="(item, key) in paramDic"
      :key="key"
    >
      <div class="titles">
        <p>{{ key }}</p>
      </div>
      <el-select
        v-model="paramDefault[key]"
        @change="changeParamSelection(key, $event)"
      >
        <el-option
          v-for="option in item"
          v-bind:key="option.value"
          v-bind:label="option.label"
          v-bind:value="option.value"
        >
        </el-option>
      </el-select>
    </el-col>
    <el-col :span="2">
      <el-button
        icon="el-icon-circle-check"
        class="el-col-select-box-button"
        @click="clickButton"
        ><br />OK!</el-button
      >
    </el-col>
  </el-row>
</template>
<script>
import axios from "axios";
import endpoint from "../util/endpoint";

export default {
  name: "CleanEmail",
  props: [
    "filenname",
    "clean_func",
    "table_cols",
    "param_dic",
    "param_default",
  ],
  data() {
    return {
      rulesForm: {},
      rules: {
        selectedCol: [
          {
            required: true,
            message: "Please select a column",
            trigger: ["blur"],
          },
        ],
      },
      cleanFunction: "",
      tableCols: [],
      paramDic: {},
      paramDefault: {},
      initColValue: "",
      selectedCol: "",
      selectedParams: {},
    };
  },
  methods: {
    changeColSelection: function (val) {
      this.selectedCol = val;
    },
    changeParamSelection: function (key, val) {
      this.selectedParams[key] = val;
    },
    clickButton: function () {
      let _this = this;
      let param = new FormData();

      param.append("clean_func", _this.cleanFunction);
      param.append("selected_col", _this.selectedCol);

      if (_this.selectedCol != "") {
        param.append("selected_params", JSON.stringify(_this.selectedParams));
        let config = {
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
        };

        axios
          .post(endpoint.API_URL + "cleanSingleCol", param, config)
          .then(function (res) {
            console.log(res);
            console.log(res.data);
            _this.$emit("newDataYes", res.data);
            _this.$emit("updateFooterLog", "yes");
          })
          .catch(function (e) {
            console.log(e);
          });
      } else {
        alert("Please select one of the columns");
      }
    },
  },
  mounted: function () {
    let _this = this;
    console.log(_this.table_cols);
    /*axios
        .post('http://localhost:7680/getFunctionParams', {
          'clean_func': _this.clean_func,
        })
        .then(function(res){
          console.log(res)
          _this.options0 = res.data.tableColumns
          //_this.list = res.data.operationLog
        })*/
    /*let _this = this
       let param = new FormData()
       param.append("filenname",_this.$props.filenname)
       let config = {
            headers: { "Content-Type": "application/x-www-form-urlencoded" }
            }
       axios.post(_this.baseServerUrl+'/csvdownload/',param, config).then(function(res){
         let initData = res.data[0]
         console.log(initData)
         Object.keys(initData).forEach(function(key){
             _this.options0.push({value:initData[key],label:initData[key]})
         })
       })*/
  },
  watch: {
    clean_func: function (val) {
      console.log(val);
      this.cleanFunction = val;
    },
    table_cols: function (val) {
      console.log(val);
      this.tableCols = val;
    },
    param_dic: function (val) {
      console.log(val);
      this.paramDic = val;
    },
    param_default: function (val) {
      console.log(val);
      this.paramDefault = val;
      this.selectedParams = val;
    },
  },
};
</script>
<style scoped>
.para-img {
  width: 40px;
  height: 50px;
}
.para-img-col {
  margin-top: 10px;
  margin-bottom: 40px;
  margin-left: 10px;
}
.el-col-select-box {
  margin-left: 10px;
  margin-bottom: 40px;
  width: 80px;
  word-break: break-all;
  word-wrap: break-word;
}
.el-col-select-box0 {
  padding-top: 20px;
  padding-left: 10px;
  margin-left: 10px;
  width: 100px;
  word-break: break-all;
  word-wrap: break-word;
}
.el-col-select-box-button {
  margin-top: 30px;
  margin-left: 40px;
}
.titles {
  height: 60px;
}
</style>
