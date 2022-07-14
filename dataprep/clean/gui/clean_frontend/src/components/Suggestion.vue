
<style scoped>
.header_center{
  display: flex;
  align-content: center;
  justify-content: center;
  border-style:solid;
  border-width:2px; 
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
  /* color: red; */
}
.new-block {
  /* background-color: red; */
  /* height: 40px; */
  border: solid;
  background-color: rgb(85, 234, 35);
  margin: 20px;
  height: 50px;
}
 .box-card {
    /* height: 340px; */
    border: groove;
}


</style>


<template>
  <el-row >
    <div  class = "header_center">
        <b class = "el-row-col">Suggestions</b>
      </div>
        <el-card class="box-card">
        <div v-for="o in 4" :key="o" class="new-block ">
            {{'Suggestion ' + o }}
        </div>
        </el-card>

  </el-row>
</template>

<script>
import axios from "axios";
import endpoint from "../util/endpoint";
import Header from './Header.vue';

export default {
  components: { Header },
  name: "Suggestion",
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
