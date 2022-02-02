import Vue from 'vue'
import Router from 'vue-router'
//import Table from '@/components/Table'
import SampleDF from '../components/SampleDF'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'SampleDF',
      component: SampleDF
    }
  ]
})
