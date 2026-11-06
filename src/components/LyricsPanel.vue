<template>
    <div class="lyrics-panel">
        <audio ref="audioEl" :src="src" preload="metadata" @timeupdate="onTimeupdate" @ended="onEnded" />

        <!-- 歌曲信息头 -->
        <header class="song-header">
            <h1 class="song-title">
                {{ title }}
                <span v-if="subtitle" class="song-subtitle">{{ subtitle }}</span>
            </h1>
            <p class="song-artists">
                <template v-for="(a, i) in artists" :key="i">
                    <a v-if="a.url" :href="a.url" target="_blank" rel="noopener" class="artist-link">{{ a.name }}</a>
                    <span v-else class="artist-name">{{ a.name }}</span>
                    <span v-if="i < artists.length - 1" class="artist-sep"> / </span>
                </template>
            </p>
        </header>

        <!-- 歌词滚动区 -->
        <div class="lyrics-scroll" ref="container" @scroll.passive="onContainerScroll">
            <ul class="lyrics-list" v-if="lines.length">
                <li v-for="(line, i) in lines" :key="i" :class="{ active: i === currentIndex }" class="lyrics-line">
                    {{ line.text }}
                </li>
            </ul>
            <div v-else class="lyrics-empty">暂无歌词</div>
        </div>

        <!-- 播放控制条 -->
        <div class="player-bar">
            <button class="play-btn" @click="togglePlay" :aria-label="playing ? '暂停' : '播放'">
                <svg v-if="!playing" viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                    <path d="M8 5.14v14l11-7z" />
                </svg>
                <svg v-else viewBox="0 0 24 24" width="18" height="18" fill="currentColor">
                    <path d="M6 5h4v14H6zM14 5h4v14h-4z" />
                </svg>
            </button>

            <div class="progress" @click="seek">
                <div class="progress-track">
                    <div class="progress-fill" :style="{ width: progress + '%' }"></div>
                </div>
            </div>

            <span class="time">{{ formatTime(currentTime) }} / {{ formatTime(duration) }}</span>
        </div>
    </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue';

const props = defineProps({
    rawLyrics: { type: String, default: '' },
    src: { type: String, default: '' },
    autoplay: { type: Boolean, default: false },
    title: { type: String, default: '' },          // 歌曲名
    subtitle: { type: String, default: '' },       // 译名（可选）
    artists: { type: Array, default: () => [] },   // [{ name, url }]
});

/* ---------- 解析 ---------- */
const TIME_TAG = /\[(\d{2}):(\d{2})\.(\d{2,3})\]/g;

const lines = computed(() => {
    if (!props.rawLyrics) return [];
    return props.rawLyrics
        .split(/\r?\n/)
        .flatMap(line => {
            const tags = [...line.matchAll(TIME_TAG)];
            if (tags.length === 0) return [];
            const text = line.replace(TIME_TAG, '').trim();
            return tags.map(m => ({
                time: Number(m[1]) * 60 + Number(m[2]) + Number(m[3]) / (m[3].length === 3 ? 1000 : 100),
                text,
            }));
        })
        .sort((a, b) => a.time - b.time);
});

/* ---------- 播放状态 ---------- */
const audioEl = ref(null);
const playing = ref(false);
const currentTime = ref(0);
const duration = ref(0);

const progress = computed(() =>
    duration.value ? Math.min((currentTime.value / duration.value) * 100, 100) : 0
);

function onTimeupdate(e) {
    currentTime.value = e.target.currentTime;
    if (e.target.duration && !isNaN(e.target.duration)) duration.value = e.target.duration;
}

function onEnded() {
    playing.value = false;
    currentTime.value = 0;
}

function togglePlay() {
    const audio = audioEl.value;
    if (!audio) return;
    if (audio.paused) {
        audio.play().then(() => (playing.value = true)).catch(err => console.warn('播放失败:', err));
    } else {
        audio.pause();
        playing.value = false;
    }
}

function seek(e) {
    const audio = audioEl.value;
    if (!audio || !duration.value) return;
    const rect = e.currentTarget.getBoundingClientRect();
    audio.currentTime = ((e.clientX - rect.left) / rect.width) * duration.value;
}

defineExpose({ play: togglePlay, audioEl });

function formatTime(sec) {
    if (!sec || isNaN(sec)) return '00:00';
    const m = Math.floor(sec / 60).toString().padStart(2, '0');
    const s = Math.floor(sec % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
}

/* ---------- 当前行 ---------- */
const currentIndex = computed(() => {
    const t = currentTime.value + 0.2;
    let idx = -1;
    for (let i = 0; i < lines.value.length; i++) {
        if (lines.value[i].time <= t) idx = i;
        else break;
    }
    return idx;
});


/* ---------- 自动滚动（scroll 事件驱动，可靠区分用户/程序滚动） ---------- */
const container = ref(null);
const lineEls = ref([]);

function setLineEl(el, i) {
    if (el) lineEls.value[i] = el;
}

function scrollToLine(idx) {
    if (idx < 0 || !container.value) return;
    const el =
        lineEls.value[idx] ??
        container.value.querySelectorAll('.lyrics-line')[idx]; // 兜底：函数 ref 失效时直接查 DOM
    if (!el) return;
    programmaticScroll = true;
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

let programmaticScroll = false; // 正在执行程序滚动
let userHold = false;           // 用户正在手动滚动
let idleTimer = null;           // 程序滚动结束判定
let resumeTimer = null;         // 用户停止后的恢复定时器

function onContainerScroll() {
    if (programmaticScroll) {
        // 平滑滚动会持续触发 scroll，等 150ms 没新事件才算程序滚动结束
        clearTimeout(idleTimer);
        idleTimer = setTimeout(() => (programmaticScroll = false), 150);
        return;
    }
    // 走到这里一定是用户在滚：暂停自动跟随，2.5 秒后回正到当前行
    userHold = true;
    clearTimeout(resumeTimer);
    resumeTimer = setTimeout(() => {
        userHold = false;
        scrollToLine(currentIndex.value);
    }, 2500);
}

watch(currentIndex, idx => {
    if (idx < 0 || userHold) return;
    nextTick(() => scrollToLine(idx));
});

watch(() => props.src, () => {
    currentTime.value = 0;
    playing.value = false;
    userHold = false;
    programmaticScroll = false;
    clearTimeout(resumeTimer);
    clearTimeout(idleTimer);
    if (container.value) container.value.scrollTo({ top: 0 });
    if (props.autoplay && audioEl.value) {
        audioEl.value.play().then(() => (playing.value = true)).catch(() => { });
    }
});
</script>

<style scoped>
.lyrics-panel {
    display: flex;
    flex-direction: column;
    height: 80vh;
    background: var(--color-background-soft, #f8f8f8);
    border: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- 歌曲信息头 ---------- */
.song-header {
    padding: 20px 24px 14px;
    border-bottom: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
    background: var(--color-background, #ffffff);
}

.song-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 24px;
    color: var(--vt-c-text-light-1, #2c3e50);
    margin: 0;
    line-height: 1.4;
}

.song-subtitle {
    font-family: 'TextRegular', sans-serif;
    font-weight: 400;
    font-size: 15px;
    color: rgb(91, 91, 91);
    /* 暗灰 */
    margin-left: 10px;
}

.song-artists {
    margin: 6px 0 0;
    font-size: 14px;
}

.artist-link {
    color: #0077ff;
    /* 你 base.css 的链接色 */
    transition: background-color 0.4s;
}

.artist-link:hover {
    background-color: var(--akt-c-yellow, rgb(255, 222, 95));
}

.artist-name {
    color: var(--vt-c-text-light-2, rgba(60, 60, 60, 0.66));
}

.artist-sep {
    color: var(--vt-c-text-light-2);
}

/* ---------- 歌词区 ---------- */
.lyrics-scroll {
    position: relative;
    flex: 1;
    min-height: 0;
    /* 关键：同上 */
    overflow-y: auto;
    scroll-behavior: smooth;
    padding: 1vh 24px;
    /* 改：45% 是按宽度算的，改用 vh */
    box-sizing: border-box;
    /* mask 等其余不变 */
}

.lyrics-scroll:active {
    cursor: grabbing;
}

.lyrics-scroll::-webkit-scrollbar {
    width: 6px;
}

.lyrics-scroll::-webkit-scrollbar-thumb {
    background: var(--vt-c-divider-light-1);
    border-radius: 3px;
}

.lyrics-list {
    list-style: none;
    margin: 0;
    padding: 0;
}

.lyrics-line {
    font-family: 'TextRegular', sans-serif;
    font-size: 16px;
    line-height: 1.9;
    padding: 4px 8px;
    text-align: center;
    /* 修复：歌词水平居中 */
    color: var(--vt-c-text-light-1, #2c3e50);
    border-radius: 6px;
    transition: opacity 0.4s, filter 0.4s, background-color 0.4s;
    cursor: default;
    will-change: opacity, filter;
}

.lyrics-line.active {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    background: var(--akt-c-yellow, rgb(255, 222, 95));
}

.lyrics-empty {
    text-align: center;
    color: var(--vt-c-text-light-2);
    padding-top: 40%;
    font-family: 'TextRegular', sans-serif;
}

/* ---------- 底部控制条 ---------- */
.player-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 12px 18px;
    border-top: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
    background: var(--color-background, #ffffff);
}

.play-btn {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 2px solid var(--vt-c-text-light-1, #2c3e50);
    color: var(--vt-c-text-light-1, #2c3e50);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background-color 0.4s, border-color 0.4s, transform 0.15s;
}

.play-btn:hover {
    background: var(--akt-c-yellow, rgb(255, 222, 95));
    border-color: var(--akt-c-yellow, rgb(255, 222, 95));
}

.play-btn:active {
    transform: scale(0.92);
}

.progress {
    flex: 1;
    cursor: pointer;
    padding: 6px 0;
}

.progress-track {
    height: 6px;
    border-radius: 3px;
    background: var(--color-border, rgba(60, 60, 60, 0.12));
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 3px;
    background: var(--akt-c-yellow, rgb(255, 222, 95));
    transition: width 0.2s linear;
}

.time {
    font-family: 'CodeRegular', monospace;
    font-size: 12px;
    color: var(--vt-c-text-light-2);
    font-variant-numeric: tabular-nums;
    flex-shrink: 0;
}
</style>