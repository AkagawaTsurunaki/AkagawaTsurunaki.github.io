<script setup lang="ts">
import { Header } from '../scripts/data';

defineProps<{
    headers: Array<Header>
}>()

function jump(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
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
    font-family: "TextBold";

}

.toc-item {
    border-left: 0px solid gray;
}
</style>