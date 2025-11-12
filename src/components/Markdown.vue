<script setup lang="ts">

import { onMounted, ref } from 'vue';
import { renderMarkdown } from '@/scripts/render/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'
import HeaderTable from './HeaderTable.vue';
import { HeaderTreeNode } from '../scripts/data';
import { parseMarkdownToc } from '@/scripts/markdownUtil';

const tableRoot = ref<HeaderTreeNode>()

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const nodes = ref<any[]>([])
const loaded = ref<boolean>()

onMounted(async () => {
  loaded.value = false
  console.log("assaaas")
  tableRoot.value = parseMarkdownToc(props.mdText)
  nodes.value = await renderMarkdown(props.mdText);
  loaded.value = true
})

</script>
<template>
  <div class="header-table">
    <HeaderTable :tree="tableRoot"></HeaderTable>
  </div>
  <div class="markdown-wrapper">
    <div v-if="!loaded" class="markdown-container">
      <MarkdownSkeleton />
    </div>
    <div v-else class="markdown-body">
      <component v-for="(n, i) in nodes" :key="i" :is="n" />
    </div>
  </div>
</template>
<style src="@/assets/markdown.css"></style>
<style scoped>
.markdown-wrapper {
  margin-left: 260px;
  padding: 24px;
  box-sizing: border-box;
}

@media (max-width: 768px) {
  .markdown-wrapper {
    margin-left: 0;
  }
}
</style>