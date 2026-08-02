import { createRouter, createWebHashHistory } from 'vue-router'
import BlogList from '@/views/BlogList.vue'
import Blog from '@/views/Blog.vue'
import Home from '@/views/Home.vue'
import NoteList from '@/views/NoteList.vue'
import Ohno from '@/views/Ohno.vue'
import PanicReport from '@/views/bonus/PanicReport.vue'

const routes = [
  { path: '/', name: 'home', component: Home },
  { path: '/blogs', name: 'blogs', component: BlogList },
  { path: '/blogs/:filePath(.*)', name: 'blogDetail', component: Blog },
  { path: '/notes', name: 'notes', component: NoteList },
  { path: '/notes/:filePath(.*)', name: 'notesDetail', component: Blog },
  // 彩蛋
  { path: '/ohno/mamiheyiwei', name: 'ohno', component: Ohno },
  { path: '/bonus/try-catch-exception/panic_report', name: 'panicReport', component: PanicReport },
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
