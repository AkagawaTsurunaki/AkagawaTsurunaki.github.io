import '@/assets/main.css'
import router from '@/scripts/router';

import { createApp } from 'vue'
import App from '@/App.vue'
import 'amfe-flexible'; // 自动调整字体
// 如果您正在使用CDN引入，请删除下面一行。
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

const app = createApp(App)
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}
app.use(router);
app.mount('#app')
