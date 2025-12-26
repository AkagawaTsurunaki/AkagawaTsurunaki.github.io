<template>
    <div class="code-wrapper">
        <div class="code-tools">
            <div class="copy-btn" @click="handleCopy">
                <Check v-if="copiedCode" class="copy-icon check-icon" />
                <CopyDocument v-else class="copy-icon" />
            </div>
            <div class="copy-btn" @click="handleCopyImg">
                <Check v-if="copiedImg" class="copy-icon check-icon" />
                <Picture v-else class="copy-icon" />
            </div>
            <span class="lang-tag">mermaid</span>
        </div>
        <pre class="code-block">
            <div ref="blockElement" class="mermaid"></div>
        </pre>
    </div>
</template>

<script setup lang="ts">
import { CopyDocument, Check, Picture } from '@element-plus/icons-vue'
import { toPng } from 'html-to-image'
import { nextTick, onMounted, ref, type Ref, type VNode, type VNodeRef } from 'vue'
import mermaid from 'mermaid';
const props = defineProps<{ language: string; code: string; rawHtml: any | null; id: string }>()
const copiedCode = ref(false)
const copiedImg = ref(false)
const blockElement = ref()
const copiedCodeTimer = ref<number>()
const copiedImgTimer = ref<number>()

onMounted(async () => {
    await nextTick()
    if (!blockElement.value) return

    try {
        mermaid.initialize({ startOnLoad: false, theme: 'default' })
        const renderId = `render-${props.id}`;
        const { svg } = await mermaid.render(renderId, props.code.trim())
        blockElement.value.innerHTML = svg
    } catch (e) {
        console.error('Mermaid rendering failure: ', e)
    }
})

async function handleCopy() {
    await handleButtonTransition(
        async () => {
            await navigator.clipboard.writeText(props.code)
        },
        copiedCode,
        copiedCodeTimer,
    )
}

async function handleCopyImg() {
    await handleButtonTransition(
        async () => {
            const katexNode = blockElement.value as HTMLElement | null
            if (!katexNode) return
            const image = await toPng(katexNode, {}).then((dataUrl) =>
                fetch(dataUrl).then((r) => r.blob()),
            )
            await navigator.clipboard.write([new ClipboardItem({ 'image/png': image })])
        },
        copiedImg,
        copiedImgTimer,
    )
}

async function handleButtonTransition(callback: Function, flag: Ref, timer: Ref) {
    if (timer) clearTimeout(timer.value) // 防止连续点击造成状态错乱
    try {
        await callback()
        flag.value = true
        timer.value = window.setTimeout(() => {
            flag.value = false
            timer.value = null
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
    opacity: 0;
    transition: opacity 0.2s ease;
}

.code-wrapper:hover .code-tools {
    opacity: 0.8;
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
    display: flex;
    flex-direction: column;
    background: #f6f8fa;
    padding: 0.8em;
    overflow-x: auto;
    border-radius: 6px;
}

.mermaid {
    text-align: center;
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
</style>
