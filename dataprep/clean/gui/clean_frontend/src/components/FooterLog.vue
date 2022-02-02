<template>
    <el-row>
        <el-col :span="4">
            <img src='../assets/footerlog.png' class="footerlog"><br>log
        </el-col>
        <el-col :span="20" class="logcontent" >
            <ol>
                <li v-for="(li,index) in list" :key="index">{{li}} </li>
            </ol>
        </el-col>
    </el-row>
</template>

<script>
import './common'
import axios from 'axios'
export default {
  name:'FooterLog',
  props:['footerLoading'],
  data(){
    return{
      list:[]
    }
  },
  mounted(){
    let _this = this
    axios.get('http://localhost:7680/getOperationLog').then(function(res){
      console.log(res.data)
      _this.list = res.data.operationLog
    })
  }
}
</script>
<style scoped>
.footerlog{
    display:inline;
    width: 45px;
    height: 40px;
    margin-top: 20px;
    margin-bottom: 10px;
}
.logcontent{
    border-left: 1px solid #ccc;
    text-align: left;
    overflow-y: scroll;
    padding: 0;
    height: 98px;
}
.logcontent::-webkit-scrollbar {
    display: none;
}
</style>
