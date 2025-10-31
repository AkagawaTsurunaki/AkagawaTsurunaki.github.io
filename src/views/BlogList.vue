<script setup lang="ts">
import { BlogItemVo } from '@/scripts/data'
import BlogItem from './BlogItem.vue'
import { onMounted, ref } from 'vue'
import { DialogLevel, openDialog } from '@/scripts/dialog'
import { getMarkdownFileInfo } from '@/scripts/markdownUtil'
import { getBlogItemList } from '@/scripts/blogRegister'

const blogItemList = ref<BlogItemVo[]>([])
let dataPreparing = ref(false)

async function loadBlogItem(blogItem: { mdFilePath: string, tags: string[], time: string }) {
  const info = await getMarkdownFileInfo(blogItem.mdFilePath)
  if (info) {
    var title = ''
    if (info?.title) {
      title = info.title
    }
    const id = blogItemList.value.length + 1
    const blog = new BlogItemVo(id, title, blogItem.tags, blogItem.time, info?.preview, blogItem.mdFilePath)
    blogItemList.value.push(blog)
  }
}

onMounted(async () => {
  dataPreparing.value = false
  try {
    const blogInfo = getBlogItemList();
    for (let index = 0; index < blogInfo.length; index++) {
      const blogItem = blogInfo[index]
      if (blogItem !== undefined) {
        await loadBlogItem(blogItem)
      }
    }
  } catch (e) {
    console.error(e)
    openDialog(
      DialogLevel.ERROR,
      '出错了',
      '获取博客列表时遇到了错误。\n刷新页面可能会修复此问题。若该问题多次出现，请联系系统管理员。',
    )
  }
  dataPreparing.value = true
})
</script>
<template>
  <ul class="blog-list-container">
    <li class="blog-item-container" v-for="bi in blogItemList" v-if="dataPreparing">
      <BlogItem :id="bi.id" :title="bi.title" :preview="bi.preview" :updatedTime="bi.updatedTime" :tags="bi.tags"
        :filePath="bi.filePath">
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
