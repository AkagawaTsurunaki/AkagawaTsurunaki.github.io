<script setup lang="ts">
import { BlogItemDto } from '@/scripts/data'
import BlogItem from './BlogItem.vue'
import BlogItemSkeleton from './../components/BlogItemSkeleton.vue'
import { onMounted, ref } from 'vue'
import { DialogLevel, openDialog } from '@/scripts/dialog'
import { getBlogItemListFromCache, getBlogItemListFromOnline } from '@/scripts/api/blogApi'
import { withTiming } from '@/scripts/diagnose/withTiming'
import { useRoute } from 'vue-router'

const blogItemList = ref<BlogItemDto[]>([])
let loaded = ref(false)
const useCache = true
const route = useRoute()


onMounted(() => {
  withTiming(
    async () => {
      loaded.value = false
      try {
        let blogInfo = []
        console.log(String(route.name))
        if (useCache) {
          blogInfo = await getBlogItemListFromCache('notes')
        }
        else {
          blogInfo = await getBlogItemListFromOnline('notes')
        }
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
          '获取笔记列表时遇到了错误。\n刷新页面可能会修复此问题。若该问题多次出现，请联系系统管理员。',
        )
      }
      loaded.value = true
    }
  )()
})
</script>
<template>
  <ul class="blog-list-container">
    <li class="blog-item-container" v-if="!loaded" v-for="_ in 3">
      <BlogItemSkeleton v-if="!loaded" />
    </li>

    <li class="blog-item-container" v-for="bi in blogItemList" v-if="loaded">
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
  padding: 0;
}

.blog-item-container {
  list-style-type: none;
}
</style>
