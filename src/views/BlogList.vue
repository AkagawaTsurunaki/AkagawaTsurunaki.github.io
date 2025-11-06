<script setup lang="ts">
import { BlogItemDto } from '@/scripts/data'
import BlogItem from './BlogItem.vue'
import { onMounted, ref } from 'vue'
import { DialogLevel, openDialog } from '@/scripts/dialog'
import { getBlogItemListFromCache } from '@/scripts/api/blogApi'

const blogItemList = ref<BlogItemDto[]>([])
let dataPreparing = ref(false)

onMounted(async () => {
  dataPreparing.value = false
  try {
    // const blogInfo = getBlogItemList();
    const blogInfo = await getBlogItemListFromCache()
    if (!blogInfo) return

    for (let index = 0; index < blogInfo.length; index++) {
      const blogItem = blogInfo[index]
      if (blogItem !== undefined) {
        blogItemList.value.push(blogItem)
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
