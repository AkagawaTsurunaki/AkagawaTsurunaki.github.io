<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { renderMarkdown } from '@/scripts/render/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'
import TableOfContents from './TableOfContents.vue';
import { Header } from '../scripts/data';
import { withTiming } from '@/scripts/diagnose/withTiming';

const headers = ref<Header[]>([])

const props = defineProps({
  mdText: { type: String, default: "帖子不见了？" }
})

const nodes = ref<any[]>([])
const displayedNodes = ref<any[]>([])
const loaded = ref(false)
const firstBatchReady = ref(false)
const BATCH_SIZE = 5

onMounted(async () => {
  loaded.value = false
  firstBatchReady.value = false
  displayedNodes.value = []

  const info = await withTiming(
    async () => await renderMarkdown(props.mdText),
    "md-renderer"
  )();

  nodes.value = info.nodes
  headers.value = info.toc
  console.log(`[诊断] VNode 总数: ${nodes.value.length}`)

  await renderInBatches()
  loaded.value = true
  console.log('[诊断] 分帧渲染结束')
})

function renderInBatches(): Promise<void> {
  return new Promise((resolve) => {
    const total = nodes.value.length
    let idx = 0
    let frame = 0

    function step() {
      frame++
      const end = Math.min(idx + BATCH_SIZE, total)
      const batch = nodes.value.slice(idx, end)

      // 关键：这里修改响应式数据
      displayedNodes.value.push(...batch)
      idx = end

      if (!firstBatchReady.value && displayedNodes.value.length > 0) {
        firstBatchReady.value = true
      }

      // 读取真实 DOM 数量
      const domCount = document.querySelectorAll('.markdown-body > *').length
      console.log(`[诊断] 第 ${frame} 帧: 挂载节点 ${idx - batch.length}~${idx}, 真实 DOM 块数: ${domCount}`)

      if (idx < total) {
        requestAnimationFrame(step)
      } else {
        console.log(`[诊断] 共 ${frame} 帧`)
        resolve()
      }
    }
    requestAnimationFrame(step)
  })
}
</script>

<<template>
  <div class="markdown-editor">
    <div class="header-table">
      <TableOfContents :headers="headers"></TableOfContents>
    </div>
    <div class="markdown-wrapper">
      <div v-if="!firstBatchReady" class="markdown-container">
        <MarkdownSkeleton />
      </div>
      <div v-show="firstBatchReady" class="markdown-body">
        <component v-for="(n, i) in displayedNodes" :key="i" :is="n" />
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