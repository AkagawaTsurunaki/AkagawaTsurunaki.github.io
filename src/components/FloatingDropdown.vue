<template>
  <div class="dropdown clickable" ref="root">
    <button @click="open = !open">
      <MoreFilled style="width: 1em; height: 1em; margin-right: 8px" />
    </button>
    <ul v-show="open" class="menu">
      <li @click="routePush('/notes')">Note</li>
      <li @click="routePush('/blogs')">Blogs</li>
      <li @click="gotoExternalSite('https://space.bilibili.com/1076299680')">Video</li>
      <li @click="routePush('/ohno/mamiheyiwei')">Arts</li>
      <li @click="routePush('/ohno/mamiheyiwei')">Papers</li>
      <li @click="routePush('/ohno/mamiheyiwei')">About</li>
    </ul>
  </div>
</template>

<script setup lang="ts">
import { routePush } from '@/scripts/router'
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { gotoExternalSite } from "@/scripts/router"

const open = ref(false)
const root = ref()

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
  left: -200%;
  top: 100%;
  margin: 4px 0 0;
  padding: 6px 0;
  min-width: 100px;
  border: 1px;
  border-radius: 4px;
  background: #fff;
  list-style: none;
  z-index: 999;
  font-family: "TextBold";
  font-size: 16px;
  box-shadow: 3px 3px 5px #dddddd;
}

.menu li {
  padding: 6px 12px;
  cursor: pointer;
  text-align: center;
}

.menu li:hover {
  background: #f5f5f5;
}
</style>
