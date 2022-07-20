
<style scoped>
.header_center{
  display: flex;
  align-content: center;
  justify-content: center;
  /* border-style:solid;
  border-width:1px;  */
}

.button_layout{
  display: flex;
  align-content: right;
  justify-content: right;
  /* border-style:solid;
  border-width:1px;  */
}
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
.el-row-col {
  line-height: 60px;
  font-size: 20px;
  font-weight: 600;
  margin: 10px;
}
.new-block {
  background-color:blanchedalmond;
  border-radius: 5px;
  margin: 5px;
  height: 100%;
}

.shadow-button {
   margin: 10px;
   box-shadow: 0 3px #999;
}
</style>


<template>
  <el-row >

    <div  class = "header_center">
      <b class = "el-row-col">Recipe</b> 
    </div>
    <div  class = "header_center">
      <el-button class="shadow-button"  type="warning" >RollBack</el-button>
      <el-button class="shadow-button"  type="primary" >Export</el-button>
    </div>
    <el-card>
      <div v-for="o in 4" :key="o" class="new-block ">
        {{'Recipe ' + o }}  
      </div>
    </el-card>
  </el-row>
</template>

<script>
import axios from "axios";
import endpoint from "../util/endpoint";

export default {

  name: "Recipe",
  data() {
    return {
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
