<script setup lang="ts">
import { onMounted, ref } from 'vue'
import Markdown from '@/components/Markdown.vue'
import { useRoute } from 'vue-router'
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'

const route = useRoute()

const blogContent = ref('正在加载...')
let loaded = ref(false)

async function setBlogDetail() {
  if (route.params.filePath && typeof route.params.filePath === 'string') {
    const res = await fetch(route.path)
    if (!res.ok) {
      console.error(`Can not fetch the resource: ${route.path}`);
      return null
    }
    blogContent.value = await res.text()
  } else {
    console.error("File path is invaliad.")
  }
}

onMounted(async () => {
  loaded.value = false
  await setBlogDetail()
  loaded.value = true
})

</script>
<template>
  <div class="markdown-container" v-if="!loaded">
    <MarkdownSkeleton></MarkdownSkeleton>
  </div>
  <div class="markdown-container" v-if="loaded">
    <Markdown :mdText="blogContent"></Markdown>
  </div>
</template>
<style>
.markdown-container {
  padding-left: 10px;
  padding-right: 10px;
}
</style>
