<script setup lang="ts">
import { Header } from '../scripts/data';
import { watch, inject, nextTick, shallowRef } from 'vue';
import { markedInstance } from '@/scripts/render/markdownRender';

const props = defineProps<{
    headers: Array<Header>
}>()

const headingHtmlMap = shallowRef<Map<string, string>>(new Map())

async function renderAll() {
    headingHtmlMap.value.clear()
    const tasks = props.headers.map(async h => ({
        id: h.id,
        html: await markedInstance.parse(h.text)
    }))
    const list = await Promise.all(tasks)
    const newMap = new Map<string, string>()
    list.forEach(({ id, html }) => newMap.set(id, html))
    headingHtmlMap.value = newMap
}
watch(() => props.headers, renderAll, { immediate: true })


const ensureChunkVisible = inject('ensureChunkVisible') as ((id: string) => void) | undefined
const getIsAccurate = inject('getIsAccurate') as ((id: string) => boolean) | undefined

function jump(id: string) {
    ensureChunkVisible?.(id)

    let attempts = 0
    const MAX_ATTEMPTS = 25
    const VIEW_ERR = 80

    const tryScroll = () => {
        const el = document.getElementById(id)
        if (!el) {
            if (attempts++ < MAX_ATTEMPTS) {
                requestAnimationFrame(tryScroll)
                return
            }
            return
        }

        const isAccurate = getIsAccurate?.(id) ?? false

        // The height of the target element is known, then just scroll into the view smoothly.
        // Otherwise, just scroll immediately with calculating the height of the targte element.
        if (isAccurate) {
            el.scrollIntoView({ behavior: 'smooth', block: 'center' })
            highlight(el)
        } else {
            const elNow = document.getElementById(id)
            if (!elNow) return
            const rect = elNow.getBoundingClientRect()
            const viewportCenter = window.innerHeight / 2

            // VIEW_ERR is the error between current postion and target position
            if (Math.abs(rect.top - viewportCenter) > VIEW_ERR) {
                elNow.scrollIntoView({ behavior: 'auto', block: 'center' })
            }

            highlight(elNow)
        }
    }

    nextTick(() => {
        requestAnimationFrame(tryScroll)
    })
}

function highlight(el: HTMLElement) {
    el.classList.add('highlight')
    setTimeout(() => el.classList.remove('highlight'), 2000)
}
</script>

<template>
    <aside class="toc" v-if="headers !== undefined">
        <ul>
            <li v-for="node in headers" :key="node.id">
                <a :style="{ paddingLeft: (node.level - 1) * 12 + 'px' }" :href="'#' + node.id"
                    @click.prevent="jump(node.id)">
                    <div class="toc-item" v-html="headingHtmlMap.get(node.id)">
                    </div>
                </a>
            </li>
        </ul>
    </aside>
</template>

<style>
.highlight {
    background: var(--akt-c-yellow);
    transition: .4s ease;
    border-radius: 6px;
}

.toc {
    position: fixed;
    left: 0px;
    top: 60px;
    bottom: 0;
    max-width: 300px;
    background: #fafafa;
    border-right: 1px solid #e5e5e5;
    overflow-y: auto;
    padding: 10px;
    font-size: 14px;
}

.toc ul {
    list-style: none;
    margin: 0;
    padding: 0;
}

.toc li a {
    display: block;
    line-height: 32px;
    color: #333;
    text-decoration: none;
    font-family: "TextBold";
}

.toc-item {
    border-left: 0px solid gray;
}
</style>