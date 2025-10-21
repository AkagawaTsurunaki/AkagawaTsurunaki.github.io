<script setup lang="ts">
import { BlogItemVo, type BlogItemDto, HttpResponseBody } from '@/scripts/data';
import BlogItem from './BlogItem.vue';
import { onMounted, ref } from 'vue';
// import { getTimeString } from '@/scripts/timeUtil';
// import axios from 'axios';
// import { plainToInstance } from 'class-transformer';
import { DialogLevel, openDialog } from '@/scripts/dialog';

const blogItemList = ref<BlogItemVo[]>([])
let dataReceived = ref(false);

async function getBlogList() {
    try {
        // const response = await axios.get<HttpResponseBody<BlogItemDto[]>>("/api/blog/list")
        // const responseBody = plainToInstance(HttpResponseBody<BlogItemDto[]>, response.data)
        // if (responseBody.isSuccess() && responseBody.data) {
        //     const result: BlogItemVo[] = []
        //     for await (const bi of responseBody.data) {
        //         const tags = bi.tags.map(tag => tag.name)
        //         const updatedTime = getTimeString(bi.updatedTime)
        //         result.push(new BlogItemVo(bi.title, bi.id, tags, updatedTime, bi.preview))
        //     }
        //     blogItemList.value = result
        // }
        throw Error("Not implemented.")
    } catch (e) {
        console.error(e)
        openDialog(DialogLevel.ERROR, "出错了", "获取博客列表时遇到了错误。\n刷新页面可能会修复此问题。若该问题多次出现，请联系系统管理员。")
    }
}

onMounted(async () => {
    dataReceived.value = false;
    await getBlogList();
    dataReceived.value = true;
})

</script>
<template>
    <ul class="blog-list-container">
        <li class="blog-item-container" v-for="bi in blogItemList" v-if="dataReceived">
            <BlogItem :id="bi.id" :title="bi.title" :preview="bi.preview" :updatedTime="bi.updatedTime" :tags="bi.tags">
            </BlogItem>
        </li>
    </ul>
</template>
<style scoped>
.blog-list-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.blog-item-container {
    list-style-type: none;
}
</style>