<template>
    <el-row>
      <div>
    <el-table
      :data="tableData"
      style="width: 100%;"
      :fit="true"
      :default-sort="{ prop: 'date', order: 'descending' }"
      :cell-style = "setColColor"
    >
      <el-table-column
        sortable
        show-overflow-tooltip
        v-for="(column, index) in tableColumns"
        v-bind:prop="column.colName"
        v-bind:label="column.colLabel"
        v-bind:key="index"
      >
      </el-table-column>
    </el-table>
     <!-- <el-button @click = "setCellColor" class="shadow-button"  type="primary" >Add</el-button> -->

  </div>
  </el-row>
</template>

<script>
import axios from "axios";

export default {
  name: "MainTable",
  data() {
    return {
      tableData: [],
      tableColumns: {},
    };
  },
  created() {
    axios
      .get("/getInitSheet")
      .then((res) => {
        this.tableData = res.data["tableData"];
        this.tableColumns = res.data["tableColumns"];
      })
      .catch((e) => {
        console.log(e);
      });
      
  },
  methods: {
    setColColor({row,column,rowIndex,columnIndex}){
      console.log(columnIndex)
      if(columnIndex===3){
        return 'background: rgb(253, 222, 218);';
      }
    },
  },
  

};
</script>
<style scoped>
</style>
