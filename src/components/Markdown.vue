<script setup lang="ts">

import { onMounted, ref } from 'vue';
import { renderMarkdown } from '@/scripts/render/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'
import TableOfContents from './TableOfContents.vue';
import { Header } from '../scripts/data';

const headers = ref<Header[]>([])

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
  const info = await renderMarkdown(props.mdText);
  nodes.value = info.nodes
  headers.value = info.toc
  loaded.value = true
})

</script>
<template>
  <div class="markdown-editor">
    <div class="header-table">
      <TableOfContents :headers="headers"></TableOfContents>
    </div>
    <div class="markdown-wrapper">
      <div v-if="!loaded" class="markdown-container">
        <MarkdownSkeleton />
      </div>
      <div v-else class="markdown-body">
        <component v-for="(n, i) in nodes" :key="i" :is="n" />
      </div>
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

  .header-table {
    display: none;
  }
}
</style>