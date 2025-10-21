<script setup lang="ts">
import { api } from '@/scripts/api';
import { type BlogDetailDto, HttpResponseBody } from '@/scripts/data';
// import { plainToInstance } from 'class-transformer';
import { onMounted, ref, watch } from 'vue';
import Markdown from '@/components/Markdown.vue';
// import { useRoute, useRouter } from 'vue-router';

const blogContent = ref("正在加载...")
let dataReceived = ref(false);

async function getBlogDetail(id: number) {
    // const response = await api.get<HttpResponseBody<BlogDetailDto>>(`/api/blog/${id}`);
    // const responseBody = plainToInstance(HttpResponseBody<BlogDetailDto>, response.data);
    // if (responseBody.isSuccess()) {
    //     if (responseBody.data) {
    //         blogContent.value = responseBody.data?.content;
    //     }
    // }
}

onMounted(async () => {
    // dataReceived.value = false;
    // const blogId = Number(useRoute().params["id"] as string)
    // if (blogId !== null) {
    //     await getBlogDetail(blogId);
    //     if (blogContent.value !== "") {
    //         dataReceived.value = true;
    //     }
    // } else {
    //     console.error("未提供博客 ID")
    // }
})

</script>
<template>
    <Markdown :mdText="blogContent" v-if="dataReceived"></Markdown>
</template>
<style></style>