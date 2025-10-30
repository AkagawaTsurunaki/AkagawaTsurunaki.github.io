<script setup lang="ts">

import { onMounted, ref, watch } from 'vue';
import { renderMarkdown } from '@/scripts/markdownRender';

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const markdownText = ref("")

onMounted(async () => {
  markdownText.value = await renderMarkdown(props.mdText);
})

</script>
<template>
  <div v-html="markdownText" class="markdown-body"></div>
</template>
<style>
.markdown-body {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.markdown-body h1 {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body h2 {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body h3 {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body h4 {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body h5 {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body strong {
  font-family: "Header Font";
  font-weight: bold;
}

.markdown-body blockquote {
  color: #777777;
  padding: 0 10px;
  border-left: solid;
  border-left-color: #dfe2e5;
  border-left-width: 5px;
}

.markdown-body blockquote .katex {
  color: #777777;
}

.katex {
  font-size: 1.1em !important;
  color: #2c3e50;
}

.katex-display {
  margin: 1.5em 0;
  padding: 1em;
  background: #f8f9fa;
  border-radius: 6px;
  overflow-x: auto;
  display: inline-flex;
}

.markdown-body pre {
  background-color: #f6f8fa;
  color: #2c3e50;
  padding: 0.2em 0.4em;
  border-radius: 3px;
  font-family: "Consolas" !important;
  font-size: 85%;
}
</style>