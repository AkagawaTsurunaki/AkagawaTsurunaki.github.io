<script setup lang="ts">
import { BlogItemVo } from '@/scripts/data'
import BlogItem from './BlogItem.vue'
import { onMounted, ref } from 'vue'
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
    const id = blogItemList.value.length + 1
    const blog = new BlogItemVo(id, title, tags, time, info?.preview, mdFilePath)
    blogItemList.value.push(blog)
  }
}

async function registerBlogList() {
  await loadBlogItem('blogs/一些证明/p.0.10.md', ['数学'], '2025-10-18 23:30')
  await loadBlogItem('blogs/一些证明/Scaled Dot-Product Attention 的公式中为什么要除以 $sqrt{d_k}$？.md', ['数学'], '2025-10-23 15:06')
  await loadBlogItem('blogs/解决方案/MikuMikuDance.md', ['MMD'], '2025-10-24 11:48')
}

onMounted(async () => {
  dataPreparing.value = false
  try {
    await registerBlogList()
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
