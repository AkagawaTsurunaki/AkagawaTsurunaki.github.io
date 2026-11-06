<template>
    <div class="lyrics-panel">
        <audio ref="audioEl" :src="src" preload="metadata" @timeupdate="onTimeupdate" @ended="onEnded" />

        <div class="lyrics-scroll" ref="container">
            <ul class="lyrics-list" v-if="lines.length">
                <li v-for="(line, i) in lines" :key="i" :class="{ active: i === currentIndex }"
                    :ref="el => setLineEl(el, i)" class="lyrics-line">
                    {{ line.text }}
                </li>
            </ul>
            <div v-else class="lyrics-empty">暂无歌词</div>
        </div>

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
    const ratio = (e.clientX - rect.left) / rect.width;
    audio.currentTime = ratio * duration.value;
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

/* ---------- 自动滚动 ---------- */
const container = ref(null);
const lineEls = ref([]);

function setLineEl(el, i) {
    if (el) lineEls.value[i] = el;
}

watch(currentIndex, async idx => {
    if (idx < 0 || !container.value) return;
    await nextTick();
    const el = lineEls.value[idx];
    if (!el) return;
    const c = container.value;
    c.scrollTo({
        top: el.offsetTop - c.clientHeight / 2 + el.clientHeight / 2,
        behavior: 'smooth',
    });
});

watch(() => props.src, () => {
    currentTime.value = 0;
    playing.value = false;
    if (container.value) container.value.scrollTo({ top: 0 });
    if (props.autoplay && audioEl.value) {
        audioEl.value.play().then(() => (playing.value = true)).catch(() => { });
    }
});
</script>

<style scoped>
/* 用你 base.css 里已定义的变量，主题切换时能自动跟着变 */
.lyrics-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--color-background-soft, #f8f8f8);
    border: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
    border-radius: 12px;
    overflow: hidden;
}

/* ---------- 歌词区 ---------- */
.lyrics-scroll {
    flex: 1;
    overflow-y: auto;
    scroll-behavior: smooth;
    padding: 45% 24px;
    box-sizing: border-box;
    mask-image: linear-gradient(to bottom,
            transparent 0,
            black 15%,
            black 85%,
            transparent 100%);
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
    line-height: 1.8;
    padding: 4px 8px;
    color: var(--vt-c-text-light-2, rgba(60, 60, 60, 0.66));
    border-radius: 6px;
    transition:
        color 0.3s,
        background-color 0.4s;
    cursor: default;
    width: fit-content;
}

/* 高亮行：你的标志性黄 + 粗体，和链接 hover 呼应 */
.lyrics-line.active {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    color: var(--vt-c-text-light-1, #2c3e50);
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
    transition:
        background-color 0.4s,
        border-color 0.4s,
        transform 0.15s;
}

.play-btn:hover {
    background: var(--akt-c-yellow, rgb(255, 222, 95));
    border-color: var(--akt-c-yellow, rgb(255, 222, 95));
    transition: 0.4s;
}

.play-btn:active {
    transform: scale(0.92);
}

/* 进度条 */
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