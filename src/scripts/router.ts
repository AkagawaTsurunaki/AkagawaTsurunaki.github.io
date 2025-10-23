import { createRouter, createWebHashHistory } from 'vue-router'
import BlogList from '@/views/BlogList.vue'
import Blog from '@/views/Blog.vue'
import Home from '@/views/Home.vue'

const routes = [
  { path: '/', name: 'home', component: Home },
  { path: '/blogs', name: 'text', component: BlogList },
  { path: '/blogs/:filePath', name: 'blogDetail', component: Blog },
]

const router = createRouter({
  history: createWebHashHistory(),
  routes: routes,
})

export function routePush(endpoint: string) {
  router.push(endpoint).catch((err) => {
    console.error('Navigation error:', err)
  })
}

export function gotoExternalSite(url: string) {
  window.open(url, '_blank')
}

export default router
