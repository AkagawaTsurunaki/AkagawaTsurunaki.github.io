<template>
    <div class="music-page">
        <MusicList class="left" :songs="songs" :selected-id="currentSong?.id" @select="onSelect" />

        <div class="right">
            <LyricsPanel v-if="currentSong" :key="currentSong.id" :raw-lyrics="lyrics" :src="currentSong.src"
                autoplay />
            <div v-else class="placeholder">从左侧选一首歌开始吧</div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, watch } from 'vue';
import MusicList from '../components/MusicList.vue';
import LyricsPanel from '../components/LyricsPanel.vue';

/* ---------- 歌单数据 ----------
 * 加歌只需：
 * 1. 把 mp3 和 lrc 丢进 public/music/
 * 2. 在下面数组加一行
 */
const songs = [
    {
        id: 1,
        title: 'Try Catch Exception （异常捕获）',
        artist: '赤川鹤鸣_P / 重音テト / 初音ミク',
        src: import.meta.env.BASE_URL + 'music/[AkagawaTsurunaki&KasaneTeto&HatsuneMiku]Try Catch Exception.mp3',
        lrc: import.meta.env.BASE_URL + 'music/lyrics/zh_CN-try_catch_exception.txt',
    },
    {
        id: 2,
        title: '（DEMO）不知道在唱些什么的铁头酱',
        artist: '赤川鹤鸣_P / 重音テト',
        src: import.meta.env.BASE_URL + 'music/[重音テト]不知道在唱些什么的铁头酱.wav',
        lrc: import.meta.env.BASE_URL + 'music/song2.lrc',
    },
];

/* ---------- 选中与歌词加载 ---------- */
const selectedId = ref(songs[0]?.id ?? null);
const currentSong = computed(() => songs.find(s => s.id === selectedId.value) ?? null);

const lyrics = ref('');

// fetch 对应的歌词文件（GitHub Pages 静态托管，同源 fetch 没问题）
watch(currentSong, async song => {
    if (!song) return;
    try {
        const res = await fetch(song.lrc);
        lyrics.value = res.ok ? await res.text() : '';
    } catch {
        lyrics.value = '';
    }
}, { immediate: true });

function onSelect(song) {
    selectedId.value = song.id;
}
</script>

<style scoped>
.music-page {
    display: grid;
    grid-template-columns: 300px 1fr;
    gap: 20px;
    width: 100%;
    max-width: 1100px;
    margin: 0 auto;
    padding: 20px;
    height: calc(100vh - 50px);
    /* 你 body 有 padding-top: 50px */
    box-sizing: border-box;
}

.left {
    height: 100%;
}

.right {
    height: 100%;
    min-width: 0;
    /* 防止 grid 子项撑破布局 */
}

.placeholder {
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'TextRegular', sans-serif;
    color: var(--vt-c-text-light-2);
    background: var(--color-background-soft);
    border: 1px dashed var(--color-border);
    border-radius: 12px;
}

/* 窄屏时改为上下布局 */
@media (max-width: 768px) {
    .music-page {
        grid-template-columns: 1fr;
        grid-template-rows: 240px 1fr;
        height: calc(100vh - 50px);
    }
}
</style>