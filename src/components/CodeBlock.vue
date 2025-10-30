<template>
  <div class="code-wrapper">
    <div class="code-tools">
      <div class="copy-btn" @click="handleCopy">
        <Check v-if="copied" class="copy-icon check-icon" />
        <CopyDocument v-else class="copy-icon" />
      </div>
      <span class="lang-tag">{{ language || 'text' }}</span>
    </div>
    <pre class="code-block"><code>{{ code }}</code></pre>
  </div>
</template>

<script setup lang="ts">
import { CopyDocument, Check } from '@element-plus/icons-vue'
import { ref } from 'vue';
const props = defineProps<{ language: string; code: string }>()
const copied = ref(false)
let timer: number | null = null

async function handleCopy() {
  if (timer) clearTimeout(timer) // 防止连续点击造成状态错乱
  try {
    await navigator.clipboard.writeText(props.code)
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
  margin: 1em 0;
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
  color: #2c3e50;
  padding: 2.2em 0.8em 0.6em;
  border-radius: 6px;
  font-family: Consolas, Menlo, monospace;
  font-size: 85%;
  overflow-x: auto;
}

.copy-icon {
  width: 1em;
  height: 1em;
  vertical-align: middle;
}

.check-icon { color: #67c23a; }
</style>
