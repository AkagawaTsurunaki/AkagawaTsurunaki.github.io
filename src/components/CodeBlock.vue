<template>
  <div class="code-wrapper">
    <div class="code-tools">
      <div class="copy-btn" @click="handleCopy">
        <Check v-if="copied" class="copy-icon check-icon" />
        <CopyDocument v-else class="copy-icon" />
      </div>
      <div class="copy-btn" @click="handleCopyImg" v-if="isKatex">
        <Check v-if="copied" class="copy-icon check-icon" />
        <Picture v-else class="copy-icon" />
      </div>
      <span class="lang-tag">{{ language || 'text' }}</span>
    </div>
    <pre v-if="!isKatex" class="code-block"><code>{{ code }}</code></pre>
    <div v-if="isKatex" class="katex code-block" v-html="rawHtml" ref="katexSvg"></div>
  </div>
</template>

<script setup lang="ts">
import { CopyDocument, Check, Picture } from '@element-plus/icons-vue'
import { toSvg } from 'html-to-image'
import { onMounted, ref } from 'vue'
const props = defineProps<{ language: string; code: string; rawHtml: any | null }>()
const copied = ref(false)
const isKatex = ref(props.language === 'katex')
const katexSvg = ref<SVGElement>()
let timer: number | null = null

onMounted(() => {
  if (props.language === 'katex') {
    console.log('katex')
  }
})

async function handleCopy() {
  await handleButtonTransition(async () => {
    await navigator.clipboard.writeText(props.code)
  })
}

async function handleCopyImg() {
  await handleButtonTransition(async () => {
    const katexNode = katexSvg.value?.querySelector('.katex') as HTMLElement | null
    if (!katexNode) return
    const image = await toSvg(katexNode).then((dataUrl) => fetch(dataUrl).then((r) => r.blob()))
    await navigator.clipboard.write([new ClipboardItem({ 'image/svg+xml': image })])
  })
}

async function handleButtonTransition(callback: Function) {
  if (timer) clearTimeout(timer) // 防止连续点击造成状态错乱
  try {
    await callback()
    copied.value = true
    timer = window.setTimeout(() => {
      copied.value = false
      timer = null
    }, 2000)
  } catch (e) {
    console.error(e)
  }
}
</script>

<style scoped>
.code-wrapper {
  position: relative;
}

.code-tools {
  position: absolute;
  top: 0;
  right: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 6px;
  font-size: 12px;
  color: #555;
  background: rgba(255, 255, 255, 0.85);
  border-bottom-left-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0);
  z-index: 1;
}

.lang-tag {
  user-select: none;
}

.copy-btn {
  padding: 2px 6px;
  border-radius: 3px;
  background: #fff;
  cursor: pointer;
  transition: border-color 0.2s;
}

.copy-btn:hover {
  border-color: #2c3e50;
}

.code-block {
  display: block;
  background: #f6f8fa;
  padding: 0.8em;
  overflow-x: auto;
  border-radius: 6px;
}

.copy-icon {
  width: 1em;
  height: 1em;
  vertical-align: middle;
}

.check-icon {
  color: #67c23a;
}

.markdown-body blockquote .katex {
  color: #777777;
}

.katex {
  font-size: 1.1em !important;
  color: #2c3e50;
}
</style>
