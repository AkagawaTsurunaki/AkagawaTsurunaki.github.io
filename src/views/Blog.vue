<script setup lang="ts">
import { onMounted, ref } from 'vue'
import Markdown from '@/components/Markdown.vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const blogContent = ref('')

onMounted(async () => {
  if (route.params.filePath && typeof route.params.filePath === 'string') {
    const res = await fetch(route.path)
    if (!res.ok) return null
    blogContent.value = await res.text()
  }
})
</script>
<template>
  <div class="markdown-container">
    <Markdown :mdText="blogContent"></Markdown>
  </div>
</template>
<style>
.markdown-container {
  padding-left: 10px;
  padding-right: 10px;
}
</style>
