import Vue from 'vue'
import Router from 'vue-router'
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
