<template>
  <div class="dropdown clickable" ref="root">
    <button @click="open = !open">
      <MoreFilled style="width: 1em; height: 1em; margin-right: 8px" />
    </button>
    <ul v-show="open" class="menu">
      <li @click="routePush('/blogs')">Blogs</li>
      <li @click="routePush('/video')">Video</li>
      <li @click="routePush('/arts')">Arts</li>
      <li @click="routePush('/papers')">Papers</li>
      <li @click="routePush('/about')">About</li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { routePush } from '@/scripts/router'
import { ref, onMounted, onBeforeUnmount } from 'vue'

const open = ref(false)
const root = ref() // 拿到整个下拉区域

/* 点击外部关闭 */
function clickOutside(e: any) {
  if (root.value && !root.value.contains(e.target)) {
    open.value = false
  }
}
onMounted(() => document.addEventListener('click', clickOutside))
onBeforeUnmount(() => document.removeEventListener('click', clickOutside))

</script>

<style scoped>
.dropdown {
  position: relative;
  display: inline-block;
}

.menu {
  position: absolute;
  left: 0;
  top: 100%;
  margin: 4px 0 0;
  padding: 6px 0;
  min-width: 120px;
  border: 1px solid #ddd;
  background: #fff;
  list-style: none;
  z-index: 999;
}

.menu li {
  padding: 6px 12px;
  cursor: pointer;
}

.menu li:hover {
  background: #f5f5f5;
}
</style>
