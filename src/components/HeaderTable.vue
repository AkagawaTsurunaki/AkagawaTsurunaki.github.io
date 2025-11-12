<script setup lang="ts">
import { HeaderTreeNode } from '../scripts/data';

defineProps<{
    tree: HeaderTreeNode
}>()

function jump(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}
</script>

<template>
    <aside class="toc" v-if="tree !== undefined">
        <ul>
            <li v-for="node in tree.children" :key="node.id">
                <a :style="{ paddingLeft: (node.level - 1) * 16 + 'px' }" :href="'#' + node.id"
                    @click.prevent="jump(node.id)">
                    {{ node.text }}
                </a>
                <TableOfContents v-if="node.children.length" :tree="node" />
            </li>
        </ul>
    </aside>
</template>

<style scoped>
.toc {
    position: fixed;
    left: 0;
    top: 60px;
    bottom: 0;
    width: 240px;
    background: #fafafa;
    border-right: 1px solid #e5e5e5;
    overflow-y: auto;
    padding: 24px 0;
    font-size: 14px;
}

ul {
    list-style: none;
    margin: 0;
    padding: 0;
}

li a {
    display: block;
    line-height: 32px;
    color: #333;
    text-decoration: none;
    transition: color .2s;
}
</style>