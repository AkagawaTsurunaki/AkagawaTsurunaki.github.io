<script setup lang="ts">
import { BlogItemVo, type BlogItemDto, HttpResponseBody } from '@/scripts/data'
import BlogItem from './BlogItem.vue'
import { onMounted, ref } from 'vue'
// import { getTimeString } from '@/scripts/timeUtil';
// import axios from 'axios';
// import { plainToInstance } from 'class-transformer';
import { DialogLevel, openDialog } from '@/scripts/dialog'
import { getMarkdownFileInfo } from '@/scripts/markdownUtil'

const blogItemList = ref<BlogItemVo[]>([])
let dataPreparing = ref(false)

async function loadBlogItem(mdFilePath: string, tags: string[], time: string) {
  const info = await getMarkdownFileInfo(mdFilePath)
  if (info) {
    var title = ''
    if (info?.title) {
      title = info.title
    }
    const blog = new BlogItemVo(1, title, tags, time, info?.preview)
    blogItemList.value.push(blog)
  }
}

async function getBlogList() {
  try {
    await loadBlogItem('/blogs/一些证明/p.0.10.md', ['数学'], '2025-10-18 23:30')
  } catch (e) {
    console.error(e)
    openDialog(
      DialogLevel.ERROR,
      '出错了',
      '获取博客列表时遇到了错误。\n刷新页面可能会修复此问题。若该问题多次出现，请联系系统管理员。',
    )
  }
}

onMounted(async () => {
  dataPreparing.value = false
  await getBlogList()
  dataPreparing.value = true
})
</script>
<template>
  <ul class="blog-list-container">
    <li class="blog-item-container" v-for="bi in blogItemList" v-if="dataPreparing">
      <BlogItem
        :id="bi.id"
        :title="bi.title"
        :preview="bi.preview"
        :updatedTime="bi.updatedTime"
        :tags="bi.tags"
      >
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
