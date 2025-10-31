<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import Markdown from '@/components/Markdown.vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const blogContent = ref('正在加载...')
let dataReceived = ref(false)

async function setBlogDetail() {
  if (route.params.filePath && typeof route.params.filePath === 'string') {
    const filePath = decodeURIComponent(route.params.filePath)
    const res = await fetch(filePath)
    if (!res.ok) return null
    blogContent.value = await res.text()
  }
}

onMounted(async () => {
  dataReceived.value = false
  await setBlogDetail()
  dataReceived.value = true
})
</script>
<template>
  <div class="markdown-container">
    <Markdown :mdText="blogContent" v-if="dataReceived"></Markdown>
  </div>
</template>
<style>
.markdown-container {
  padding-left: 10px;
  padding-right: 10px;
}
</style>
