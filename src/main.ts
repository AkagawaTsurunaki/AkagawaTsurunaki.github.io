import '@/assets/main.css'
import router from '@/scripts/router';

import { createApp } from 'vue'
import App from '@/App.vue'
import 'amfe-flexible'; // 自动调整字体

const app = createApp(App)
app.use(router);
app.mount('#app')
