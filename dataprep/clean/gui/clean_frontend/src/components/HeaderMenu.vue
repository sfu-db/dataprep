<template>
  <el-table-column
    :prop="propName"
    :label="menuText"
    :width="width">
    <template slot="header" slot-scope="scope">
      <el-dropdown trigger="click" size="medium " @command="handleCommand">
        <span style="color:#909399">
          {{ menuText }}<i class="el-icon-arrow-down el-icon--right" />
        </span>
        <el-dropdown-menu slot="dropdown">
          <el-dropdown-item
            v-for="(item, index) in section"
            v-bind:key="index"
            v-bind:command="item"
          >
            {{ item }}
          </el-dropdown-item>
        </el-dropdown-menu>
      </el-dropdown>
    </template>
  </el-table-column>
</template>

<script>
import axios from "axios"

export default {
  name: "HeaderMenu",
  props:{
    propName: {
      type: String,
      default: ''
    },
    menuText: {
      type: String,
      default: ''
    },
    width: {
      type: Number,
      default: 100
    },
    section: {
      type: Array,
      default: [
        'Option 1',
        'Option 2'
      ]
    }
  },
  data() {
    return {
      cleanedData: {}
    }
  },
  methods:{
    handleCommand(item) { //这里定义方法可在选取时触发
      //this.menuText = item
      axios.post('http://localhost:7680/cleanData', {
        'clean_func': item,
        'col': this.menuText
      }).then(res => {
          console.log(res)
          this.cleanedData = res.data
          this.$emit('getCleanedData', this.cleanedData)
      }).catch(e => {
          console.log(e)
      })
    }
  }
}
</script>

<style scoped>

</style>
