<template>
        <el-row justify="center" align="middle">
            <el-col :span="6" >
                <el-button  icon="el-icon-s-home" class="header_button" @click="pageToOrign"><br>origin</el-button>
                <el-button  icon="el-icon-back" class="header_button" @click="clickUndo"><br>undo</el-button>
                <el-button  icon="el-icon-right" class="header_button" @click="clickRedo"><br>redo</el-button>
            </el-col>
            <el-col :span="11"  class="el-row-col">DataPrep.Clean UI</el-col>
            <!--<el-col :span="3" >
              <el-upload
                action=""
                accept="csv"
                :http-request="submitUpload"
                multiple
                show-file-list="false">
                <el-button  icon="el-icon-upload2"><br>Import</el-button>
              </el-upload>
            </el-col>-->
          <el-col :span="7">
            <el-button  icon="el-icon-download" class="header_button" @click="clickExportCSV"><br>CSV</el-button>
            <el-button  icon="el-icon-download" class="header_button" @click="clickExportDF"><br>DataFrame</el-button>
            <el-button  icon="el-icon-download" class="header_button" @click="clickExportLog"><br>Log</el-button>
          </el-col>
        </el-row>
</template>

<script>
import './common'
import axios from 'axios'

export default {
  name: 'Header',
  data () {
    return {
      sample_df:'sample_df',
      exportOptions: [
        {
          value: 'CSV',
          label: 'CSV'
        },
        {
          value: 'DataFrame',
          label: 'DataFrame'
        },
      ],
      exportValue: '',
    }
  },
  methods:{
    pageToOrign:function(){
      let _this = this
      axios
        .get("http://localhost:7680/getOriginData")
        .then(function(res){
          console.log(res);
          _this.$emit('newDataYes', res.data);
          _this.$emit('updateFooterLog', 'yes')
        })
        .catch(function(e){console.log(e)});
    },
    submitUpload(item){
      console.log(item);
      let _this = this;
      if (item) {
        let param = new FormData();

        param.append("file", item.file);
        for (let key of param.entries()) {
          console.log(key[0] + ', ' + key[1]);
        }
        let config = {
          headers: { "Content-Type": "multipart/form-data" }
        };
        axios
          .post("http://localhost:7680/fileUpload", param, config)
          .then(function(res){
            console.log(res);
            _this.$emit('newDataYes','yes');
            _this.sample_df = item.file.name;
          })
          .catch(function(e){console.log(e)});
      }
      else{
      }
    },
    clickUndo:function(){
      let _this = this
      axios.get("http://localhost:7680/undo").then(function(res){
        console.log(res.data)
        _this.$emit('newDataYes','yes')
        _this.$emit('updateFooterLog','yes')
      })
    },
    clickRedo:function(){
      let _this = this
      axios.get("http://localhost:7680/redo").then(function(res){
        console.log(res.data)
        _this.$emit('newDataYes','yes')
        _this.$emit('updateFooterLog','yes')
      })
    },
    clickExportCSV: function (val){
      axios.post("http://localhost:7680/exportCSV")
        .then((res) => {
          let data = res.data;
          const blob = new Blob([data], { type: 'application/csv' })
          let link = document.createElement('a')
          link.href = window.URL.createObjectURL(blob)
          link.download = 'cleaned_df.csv'
          link.click()
        })
        .catch(error => {
          console.error(error);
        });
    },
    clickExportDF: function (val){
      console.log(val)
      axios.get("http://localhost:7680/exportDF").then(function(res){
       console.log(res)
      })
    },
    clickExportLog: function (val){
      console.log(val)
      axios.get("http://localhost:7680/exportExecutionLog").then(function(res){
        console.log(res)
      })
    },
  },
  watch:{
    exportValue: function (val){
      console.log(val)

    }
  }
}
</script>

<style scoped>
.el-row-col{
    line-height: 60px;
    font-size: 25px;
    font-weight: 600;
}
.header_button{
    font-size: 18px;
    padding:3px;
    margin-top:8px;
    border: 1px solid #B3C0D1;
    border-radius: 5px;
}
.upload-demo{
  display: none;
}
</style>
