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
<style>
.markdown-body {
  display: flex;
  flex-direction: column;
  gap: 5px;
}

.markdown-body h1 {
  font-family: "TextBold";
  font-weight: bold;
}

.markdown-body h2 {
  font-family: "TextBold";
  font-weight: bold;
}

.markdown-body h3 {
  font-family: "TextBold";
  font-weight: bold;
}

.markdown-body h4 {
  font-family: "TextBold";
  font-weight: bold;
}

.markdown-body h5 {
  font-family: "TextBold";
  font-weight: bold;
}

.markdown-body strong {
  font-family: "TextBold";
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

.katex-display {
  margin: 0 !important;
}
</style>