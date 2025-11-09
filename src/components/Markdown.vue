<script setup lang="ts">

import { onMounted, ref, watch } from 'vue';
import { renderMarkdown } from '@/scripts/markdownRender';

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const nodes = ref<any[]>([]);

onMounted(async () => {
  nodes.value = await renderMarkdown(props.mdText);
})

</script>
<template>
  <div class="markdown-body">
    <!-- 直接渲染 VNode 数组 -->
    <component v-for="(node, i) in nodes" :key="i" :is="node" />
  </div>
</template>
<style src="@/assets/markdown.css"></style>