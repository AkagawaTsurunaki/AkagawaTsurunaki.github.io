<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import Markdown from '@/components/Markdown.vue'
// import { useRoute, useRouter } from 'vue-router';

const props = defineProps({
  filePath: {
    type: String,
    default: null,
  },
})
// const filePath = ref('blogs/一些证明/p.0.10.md')
const blogContent = ref('正在加载...')
let dataReceived = ref(false)


async function setBlogDetail() {
  const res = await fetch(props.filePath)
  if (!res.ok) return null
  blogContent.value = await res.text()
}

onMounted(async () => {
  dataReceived.value = false
  await setBlogDetail()
  dataReceived.value = true
})
</script>
<template>
  <Markdown :mdText="blogContent" v-if="dataReceived"></Markdown>
</template>
<style></style>
