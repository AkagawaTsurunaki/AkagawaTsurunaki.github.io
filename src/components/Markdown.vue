<script setup lang="ts">

import { onMounted, ref, watch } from 'vue';
import { renderMarkdown } from '@/scripts/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const nodes = ref<any[]>([]);
const isRendered = ref<boolean>(false)

watch(() => props.mdText, (_: any) => {
  render()
}, { immediate: true }) 

async function render() {
  isRendered.value = false
  if (props.mdText) {
    console.log(`Markdown content: \n${props.mdText}`)
  }
  nodes.value = await renderMarkdown(props.mdText);
  console.log(`Markdown rendering finished: ${nodes.value.length} rendered.`)
  isRendered.value = true
}

</script>
<template>
  <div class="markdown-container" v-if="!isRendered">
    <MarkdownSkeleton></MarkdownSkeleton>
  </div>
  <div class="markdown-body" v-if="isRendered">
    <!-- 直接渲染 VNode 数组 -->
    <component v-for="(node, i) in nodes" :key="i" :is="node" />
  </div>
</template>
<style src="@/assets/markdown.css"></style>