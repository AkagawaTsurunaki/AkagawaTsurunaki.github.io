<script setup lang="ts">
import avartar from '@/assets/avartar.png'
import { routePush } from '@/scripts/router'
import FloatingDropdown from './FloatingDropdown.vue';
import { gotoExternalSite } from "@/scripts/router"
</script>
<template>
  <header class="navbar">
    <div class="navbar-item">
      <div class="nav-avartar" @click="routePush('/')">
        <img :src="avartar" alt="" width="30" />
        <div class="header-font">AkagawaTsurunaki</div>
      </div>
    </div>
    <div class="navbar-item" id="navbar-button-group">
      <ul class="navbar-button-group">
        <li @click="routePush('/notes')">
          <a class="navbar-a header-font">Notes</a>
        </li>
        <li @click="routePush('/blogs')">
          <a class="navbar-a header-font">Blogs</a>
        </li>
        <li @click="gotoExternalSite('https://space.bilibili.com/1076299680')">
          <a class="navbar-a header-font">Video</a>
        </li>
        <li @click="routePush('/ohno/mamiheyiwei')">
          <a class="navbar-a header-font">Arts</a>
        </li>
        <li @click="routePush('/ohno/mamiheyiwei')">
          <a class="navbar-a header-font">Papers</a>
        </li>
        <li @click="routePush('/ohno/mamiheyiwei')">
          <a class="navbar-a header-font">About</a>
        </li>
      </ul>
    </div>
    <div class="navbar-item" id="hamberger">
      <FloatingDropdown></FloatingDropdown>
    </div>
  </header>
  <div class="nav-placeholder"></div>
</template>
<style scoped>
.nav-placeholder {
  height: 60px;
  /* 与导航栏同高即可 */
}

#hamberger {
  display: none;
}

@media (max-width: 768px) {
  #navbar-button-group {
    display: none !important;
  }

  #hamberger {
    display: flex !important;
  }
}

.navbar {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: space-between;

  background: #fcfcfc;
  box-shadow: 3px 3px 5px #dddddd;
  height: 60px;
  padding: 0 5%;
  z-index: 10;
}

.navbar-item {
  max-width: 400px;
}

.navbar-button-group {
  display: flex;
  list-style: none;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: center;
  align-items: center;
}

.navbar-button-group li {
  cursor: pointer;
  height: 60px;
  display: flex;
  align-items: center;
}

.nav-avartar {
  display: flex;
  font-family: 'TextBold', sans-serif;
  font-size: 18px;
  font-weight: 500;
  gap: 10px;
}

.search-bar {
  min-width: 400px;
  height: 30px;
  border: 1;
  border-radius: 20px;

  font-family: 'TextRegular';
}

.header-font {
  font-family: 'TextBold', sans-serif;
  font-size: 18px;
  font-weight: 500;
}

.navbar-a {
  color: #2c3e50;
  transition: 0.4s;
  display: flex;
  align-items: center;
  height: 100%;
  padding: 0 10px;
}

.navbar-a:hover {
  background-color: rgb(255, 222, 95);
}
</style>
