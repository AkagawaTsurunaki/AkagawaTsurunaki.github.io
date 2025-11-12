<script setup lang="ts">
import Tag from '@/components/Tag.vue';
import { renderMarkdown } from '@/scripts/render/markdownRender';
import { routePush } from '@/scripts/router';
import { getTimeString } from '@/scripts/timeUtil';
import { onMounted, ref, type PropType, type VNode } from 'vue';
import BlogItemSkeleton from '../components/BlogItemSkeleton.vue'

const props = defineProps({
    id: {
        type: Number,
        default: null
    },
    title: {
        type: String,
        default: "请输入标题"
    },
    preview: {
        type: String,
        default: "请输入文字。"
    },
    updatedTime: {
        type: String,
        default: getTimeString()
    },
    tags: {
        type: Array as PropType<String[]>,
        default: []
    },
    filePath: {
        type: String,
        default: null
    }
})
const mdTitle = ref<VNode[]>()
const mdPreview = ref<VNode[]>()
let loaded = ref(false)

onMounted(async () => {
    loaded.value = false
    mdTitle.value = await renderMarkdown(props.title)
    mdPreview.value = await renderMarkdown(props.preview, ['image'])
    removeSpecialToken()
    loaded.value = true
})

function removeSpecialToken() {
    const previewElmList = document.getElementsByClassName("markdown-body");
    for (const elm of previewElmList) {
        removeHr(elm)
        replaceHeader(elm)
    }
}

function removeHr(elm: Element) {
    const hrs = elm.querySelectorAll('hr')
    hrs.forEach((hr) => {
        hr.remove();
    });
}

function replaceHeader(elm: Element) {
    for (let i = 1; i < 6; i++) {
        const hs = elm.querySelectorAll(`h${i}`)
        hs.forEach((h) => {
            h.replaceWith(h.textContent);
        });
    }
}

function gotoBlog() {
    if (props.id !== null && props.filePath !== null) {
        // routePush(`/blogs/${encodeURIComponent(props.filePath)}`)
        const endpoints = props.filePath.split("/")
        const folder = endpoints[0] 
        const path = endpoints[1]
        routePush(props.filePath)
    } else {
        console.error("未提供博客地址")
    }
}

</script>
<template>
    <BlogItemSkeleton v-if="!loaded" />
    <div @click="gotoBlog" class="blog-item-container" v-if="loaded">
        <div>
            <h2 class="blog-title">
                <component v-for="(node, i) in mdTitle" :key="i" :is="node" />
            </h2>
        </div>
        <!-- 直接渲染 VNode 数组 -->
        <component class="markdown-body" v-for="(node, i) in mdPreview" :key="i" :is="node" />
        <div class="split"></div>
        <div class="extra-info-container">
            <div class="updated-time-container">{{ updatedTime }}</div>
            <div class="tags-container">
                <ul v-for="tagString in props.tags">
                    <Tag :tag="`${tagString}`"></Tag>
                </ul>
            </div>
        </div>
    </div>

</template>
<style scoped>
.split {
    height: 10px;
}

.blog-item-container {
    display: flex;
    flex-direction: column;
    padding: 10px;

    border-radius: 10px;
    box-shadow: 1px 1px 10px rgba(0, 0, 0, 0.1);
}

.blog-title {
    font-family: "TextBold";
    font-size: 24px;
}

.tags-container ul {
    display: inline-flex;
    flex-direction: row;
}

.extra-info-container {
    display: inline-flex;
    justify-content: space-between;
    font-size: 16px;
}

.blog-item-container:hover {
    cursor: pointer;
    background-color: #fff1c8;
    transition: 0.4s;
    border-radius: 10px;
}

.updated-time-container {
    font-family: "TextRegular";
    color: grey;
}
</style>