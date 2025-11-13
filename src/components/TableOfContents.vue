<script setup lang="ts">
import { Header } from '../scripts/data';

defineProps<{
    headers: Array<Header>
}>()


function jump(id: string) {
    const el = document.getElementById(id);
    if (!el) return;
    el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    const onScrollEnd = () => {
        el.classList.add('highlight');
        const removeHighlight = () => {
            el.classList.remove('highlight');
            document.removeEventListener('scroll', removeHighlight);
        };
        document.addEventListener('scroll', removeHighlight, { once: true });
    };
    document.addEventListener('scrollend', onScrollEnd, { once: true });
    setTimeout(onScrollEnd, 1000);
}
</script>

<template>
    <aside class="toc" v-if="headers !== undefined">
        <ul>
            <li v-for="node in headers" :key="node.id">
                <a :style="{ paddingLeft: (node.level - 1) * 12 + 'px' }" :href="'#' + node.id"
                    @click.prevent="jump(node.id)">
                    <div class="toc-item">
                        {{ node.text }}
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