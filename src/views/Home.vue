<script setup lang="ts">
import avartar from "@/assets/avartar.png"
import bilibiliIcon from "@/assets/icons/bilibilidonghua.svg"
import githubIcon from "@/assets/icons/github.svg"
import emailIcon from "@/assets/icons/email.svg"
import pixivIcon from "@/assets/icons/pixiv.svg"
import { gotoExternalSite } from "@/scripts/router"

function sendEmail() {
    const email = 'AkagawaTsurunaki@outlook.com';
    const mailtoLink = `mailto:${email}`;
    window.location.href = mailtoLink;
}

</script>
<template>
    <div class="introduction">
        <div><img :src="avartar" alt="" width="100"></div>
        <div class="name">赤川鹤鸣 / AkagawaTsurunaki</div>
        <div>一名编程、剪辑、绘画、小说啥都搞的程序员。</div>
        <div class="link-buttons">
            <button class="link-button" @click="gotoExternalSite('https://github.com/AkagawaTsurunaki')">
                <img class="link-icon" :src="githubIcon" alt="">
                <span>Github</span>
            </button>
            <button class="link-button" @click="gotoExternalSite('https://space.bilibili.com/1076299680')">
                <img class="link-icon" :src="bilibiliIcon" alt="">
                <span>Bilibili</span>
            </button>
            <button class="link-button" @click="gotoExternalSite('https://www.pixiv.net/users/102518593')">
                <img class="link-icon" :src="pixivIcon" alt="">
                <span>Pixiv</span>
            </button>
            <button class="link-button" @click="sendEmail">
                <img class="link-icon" :src="emailIcon" alt="">
                <span>Email</span>
            </button>
        </div>
        <div class="gradient-underline"></div>
    </div>
</template>
<style scoped>
.gradient-underline {
    width: 200px;
    /* 下划线宽度 */
    height: 3px;
    /* 下划线粗细 */
    background: linear-gradient(to left, #ff4d4c, #ffd34f, #ff4d4c);
    background-size: 200% 100%;
    /* 扩展渐变范围 */
    animation: flow 3s infinite linear;
    border-radius: 10px;
}

@keyframes flow {
    0% {
        background-position: 0% 50%;
    }

    100% {
        background-position: 200% 50%;
    }
}

.introduction {
    font-family: "Common Font";
    display: flex;
    flex-direction: column;
    align-items: center;
}

.introduction>* {
    margin: 5px;
}

.introduction .name {
    font-family: "Header Font";
    font-size: 20px;
}

.link-button {
    border: none;
    border-radius: 30px;
    align-items: center;
    font-family: "Common Font";
    gap: 8px;
    display: inline-flex;
}

.link-icon {
    flex-direction: row;
    width: 15px;
}

.link-icon>* {
    gap: 10px;
}

.link-buttons {
    display: flex;
    gap: 15px;
}
</style>