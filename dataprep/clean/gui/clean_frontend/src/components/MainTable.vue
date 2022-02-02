<template>
    <el-table
      :data="tableData"
      style="width: 100%"
      border
      :fit='true'
      :default-sort="{prop: 'date', order: 'descending'}"
    >
      <el-table-column sortable
                       show-overflow-tooltip
                       v-for="(column, index) in tableColumns"
                       v-bind:prop="column.colName"
                       v-bind:label="column.colLabel"
                       v-bind:key="index"
      >
      </el-table-column>
    </el-table>
</template>

<script>
import './common'
import axios from 'axios'

export default {
  name:'MainTable',
  data(){
    return{
      tableData:[],
      tableColumns:{}
    }
  },
  created() {
    axios.get('/getInitSheet').then(res => {
      //console.log(res);
      this.tableData = res.data['tableData'];
      this.tableColumns = res.data['tableColumns'];
    }).catch(e => {
      console.log(e)
    });
  }
}
</script>
<style scoped>

</style>
