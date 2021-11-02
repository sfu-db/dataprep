<template>
  <el-table
    :data="tableData"
    style="width: 100%"
  >
    <header-menu v-for="(column, index) in tableColumns"
                 v-bind:key="index"
                 v-bind:prop-name="column.colName"
                 v-bind:menu-text="column.colLabel"
                 v-bind:section="cleanFunctionList"
                 v-bind:width="column.colWidth"
                 v-on:getCleanedData="updateCleanedData"
    >
    </header-menu>
  </el-table>
</template>

<script>
import axios from "axios"
import HeaderMenu from "./HeaderMenu"

export default {
  name: "Table",
  components: {
    HeaderMenu
  },
  data() {
    return {
      tableData: [],
      tableColumns: [],
      cleanFunctionList: [
        'clean_email',
        'clean_address',
        'clean_headers',
        'clean_country',
        'clean_date',
        'clean_duplication',
        'clean_lat_long',
        'clean_ip',
        'clean_phone',
        'clean_text',
        'clean_url',
        'clean_df'
      ],
    }
  },
  created() {
    axios.get('http://localhost:7680/getInitSheet').then(res => {
      console.log(res);
      this.tableData = res.data['tableData'];
      this.tableColumns = res.data['tableColumns'];
    }).catch(e => {
      console.log(e)
    });
  },
  methods:{
    updateCleanedData(cleaned_data){
      this.tableData = cleaned_data['tableData'];
      this.tableColumns = cleaned_data['tableColumns'];
    }
  }
}
</script>
