<script setup lang="ts">
import avartar from "@/assets/avartar.png"
import searchIcon from "@/assets/icons/search.svg";
import { routePush } from '@/scripts/router';

</script>
<template>
    <div class="navbar">
        <div class="navbar-item">
            <div class="nav-avartar">
                <img :src="avartar" alt="" width="30">
                <div class="header-font">AkagawaTsurunaki</div>
            </div>
        </div>
        <div class="navbar-item" style="position: relative;">
            <img :src="searchIcon" alt="" width="20"
                style="position: absolute; left: 3%; top: 50%; transform: translateY(-50%);">
            <input class="search-bar" type="text" placeholder="Search..." style="padding-left: 35px;">
        </div>
        <div class="navbar-item">
            <ul class="navbar-button-group">
                <li @click="routePush('/blog')">
                    <a class="navbar-a header-font">Blogs</a>
                </li>
                <li @click="routePush('/video')">
                    <a class="navbar-a header-font">Video</a>
                </li>
                <li @click="routePush('/arts')">
                    <a class="navbar-a header-font">Arts</a>
                </li>
                <li @click="routePush('/papers')">
                    <a class="navbar-a header-font">Papers</a>
                </li>
                <li @click="routePush('/about')">
                    <a class="navbar-a header-font">About</a>
                </li>
            </ul>
        </div>
    </div>

</template>
<style scoped>
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
    padding: 0 50px;
    z-index: 10;
}

.navbar-item {
    min-width: 400px;
}

.navbar-button-group {
    display: flex;
    list-style: none;
}

.navbar-button-group li {
    cursor: pointer;
    height: 60px;
    display: flex;
    align-items: center;
}

.nav-avartar {
    display: flex;
    font-family: 'Header Font', sans-serif;
    font-size: 18px;
    font-weight: 500;
    gap: 10px;
}

.search-bar {
    min-width: 400px;
    height: 30px;
    border: 1;
    border-radius: 20px;

    font-family: "Common Font";
}

@font-face {
    font-family: 'Header Font';
    src: url('src/assets/fonts/HarmonyOS_Sans_SC_Black.ttf') format('truetype');
    font-weight: 900;
    /* Black 字重 */
}

.header-font {
    font-family: 'Header Font', sans-serif;
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