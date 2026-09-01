<script setup lang="ts">
import { onMounted, ref, shallowRef, nextTick, provide, type CSSProperties } from 'vue';
import { renderMarkdown } from '@/scripts/render/markdownRender';
import MarkdownSkeleton from '@/components/MarkdownSkeleton.vue'
import TableOfContents from './TableOfContents.vue';
import { Header } from '../scripts/data';
import { withTiming } from '@/scripts/diagnose/withTiming';

const headers = shallowRef<Header[]>([])

const props = defineProps({
  mdText: {
    type: String,
    default: "帖子不见了？"
  }
})

const nodes = shallowRef<any[]>([])
const loaded = ref(false)

// Chunked viewpoint loading
// Each chunk has properties: index (number), visible, height, HTMLelement
const CHUNK_SIZE = 20
const chunks = shallowRef<any[][]>([])
const chunkVisible = ref<boolean[]>([])
const chunkHeights = ref<Record<number, number>>({})
const chunkElements = ref<Map<number, HTMLElement>>(new Map())
const headingNodeIndices = shallowRef<number[]>([])

let observer: IntersectionObserver | null = null

onMounted(async () => {
  loaded.value = false
  const info = await withTiming(async () => {
    return await renderMarkdown(props.mdText)
  }, "md-renderer")();

  nodes.value = info.nodes
  headers.value = info.toc
  headingNodeIndices.value = info.headingNodeIndices || [] // The heading positions in the page.

  // Chunked process elements
  const chunksTmp: any[][] = []
  for (let i = 0; i < nodes.value.length; i += CHUNK_SIZE) {
    chunksTmp.push(nodes.value.slice(i, i + CHUNK_SIZE))
  }
  chunks.value = chunksTmp
  chunkVisible.value = new Array(chunksTmp.length).fill(false)

  loaded.value = true

  await nextTick()
  setupObserver()
})

// Chunked loading can break the proper functionality of the scrollbar, 
// so a "preloading" mechanism is implemented by triggering a callback when an element is still 800px away from the viewport.
// When an element enters or leaves the preloading viewport range, mark the `visible` property of that chunk as true or false.
function setupObserver() {
  observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      const idx = Number((entry.target as HTMLElement).dataset.chunkIndex)
      if (entry.isIntersecting) {
        if (!chunkVisible.value[idx]) {
          chunkVisible.value[idx] = true
        }
      } else {
        if (chunkHeights.value[idx] !== undefined && chunkVisible.value[idx]) {
          chunkVisible.value[idx] = false
        }
      }
    })
  }, {
    rootMargin: '800px 0px',
    threshold: 0
  })

  chunkElements.value.forEach((el, i) => {
    if (el) {
      el.dataset.chunkIndex = String(i)
      if (observer) {
        observer.observe(el)
      }
    }
  })
}

// Vue
function setChunkRef(el: HTMLElement | null, index: number) {
  if (el) {
    chunkElements.value.set(index, el)
    el.dataset.chunkIndex = String(index)
    observer?.observe(el)
  } else {
    // Clean unloaded element
    const oldEl = chunkElements.value.get(index)
    if (oldEl && observer) observer.unobserve(oldEl)
    chunkElements.value.delete(index)
  }
}

function recordHeight(el: HTMLElement | null, index: number) {
  if (el && chunkVisible.value[index]) {
    chunkHeights.value[index] = el.offsetHeight
  }
}

function getPlaceholderStyle(index: number): CSSProperties {
  const h = chunkHeights.value[index]
  return {
    height: h ? `${h}px` : '300px',
    background: '#f0f0f0',
    borderRadius: '4px',
    margin: '8px 0',
    position: 'relative',
    overflow: 'hidden'
  }
}

provide('ensureChunkVisible', (id: string) => {
  const headerIndex = headers.value.findIndex(h => h.id === id)
  if (headerIndex < 0 || headingNodeIndices.value[headerIndex] === undefined) return

  const nodeIdx = headingNodeIndices.value[headerIndex]
  const targetChunkIdx = Math.floor(nodeIdx / CHUNK_SIZE)

  // Only render the target chunk and its neighboring chunks, up to 5 chunks in total, 
  // and never render the intermediate path chunks (skip).
  for (let i = Math.max(0, targetChunkIdx - 2); i <= Math.min(chunks.value.length - 1, targetChunkIdx + 2); i++) {
    chunkVisible.value[i] = true
  }
})

// For the content menu jump feature: 
// Determine whether the chunk heights from the beginning of the document to the target are all known. 
// If all are known, it means the placeholder heights equal the actual heights, and the `offsetTop` is accurate.
provide('getIsAccurate', (id: string): boolean => {
  const headerIndex = headers.value.findIndex(h => h.id === id)
  if (headerIndex < 0 || headingNodeIndices.value[headerIndex] === undefined) return false

  const targetNodeIdx = headingNodeIndices.value[headerIndex]
  const targetChunkIdx = Math.floor(targetNodeIdx / CHUNK_SIZE)

  for (let i = 0; i <= targetChunkIdx; i++) {
    if (chunkHeights.value[i] === undefined) return false
  }
  return true
})
</script>

<<template>
  <div class="markdown-editor">
    <div class="header-table">
      <TableOfContents :headers="headers"></TableOfContents>
    </div>
    <div class="markdown-wrapper">
      <div v-if="!loaded" class="markdown-container">
        <MarkdownSkeleton />
      </div>
      <div v-else class="markdown-body">
        <div v-for="(chunk, i) in chunks" :key="i" :ref="(el: any) => setChunkRef(el, i)">
          <div v-if="chunkVisible[i]" :ref="(el: any) => recordHeight(el, i)">
            <component v-for="(n, j) in chunk" :key="j" :is="n" />
          </div>
          <div v-else :style="getPlaceholderStyle(i)"></div>
        </div>
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