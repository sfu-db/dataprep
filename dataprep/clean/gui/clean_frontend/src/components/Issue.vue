
<style scoped>
.header_center{
  display: flex;
  align-content: center;
  justify-content: center;
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
  background-color:rgb(244, 193, 187);
  margin: 5px;
  height: 50px;
}

.whole-block {
  border-radius: 5px;
  margin: 10px;
}
 
</style>


<template>
  <el-row >
    <div class = 'whole-block'>
      
      <el-card class="shadow">
          
                <div  class = "header_center">
                  <b class = "el-row-col">Issues</b>
              </div>
          <div class="new-block">
            Missing value in 
            <b> email </b>
        </div>
         <div class="new-block">
           Missing format in 
            <b> city of birth  </b>
        </div>
         <div class="new-block">
           Duplicate values in 
            <b> phone number </b>
        </div>
         <div class="new-block">
           Invalid value in 
            <b> email </b>
        </div>
        </el-card>
           </div>
  </el-row>
</template>

<script>
import axios from "axios";
import endpoint from "../util/endpoint";
import Header from './Header.vue';

export default {
  components: { Header },
  name: "Issue",
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
        axios.post(endpoint.API_URL + "getFunctionParams", {
            clean_func: nval,
          }).then(function (res) {
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
