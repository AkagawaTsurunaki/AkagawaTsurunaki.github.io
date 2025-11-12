<script setup lang="ts">

import { onMounted, ref } from 'vue';
import { renderMarkdown } from '@/scripts/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const nodes = ref<any[]>([]);
const loaded = ref<boolean>(false)

onMounted(async () => {
  loaded.value = false
  nodes.value = await renderMarkdown(props.mdText);
  loaded.value = true
})

</script>
<template>
  <div class="markdown-container" v-if="!loaded">
    <MarkdownSkeleton></MarkdownSkeleton>
  </div>
  <div class="markdown-body" v-if="loaded">
    <!-- 直接渲染 VNode 数组 -->
    <component v-for="(node, i) in nodes" :key="i" :is="node" />
  </div>
</template>
<style src="@/assets/markdown.css"></style>