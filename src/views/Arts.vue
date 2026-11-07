<template>
    <div class="art-page">
        <h1 class="page-title">画作</h1>

        <div class="gallery" ref="galleryEl">
            <div v-for="(col, ci) in columns" :key="ci" class="gallery-col">
                <figure v-for="art in col" :key="art.src" class="art-card" @click="openViewer(art)">
                    <div class="img-wrapper">
                        <img :src="art.src" :alt="art.title" loading="lazy" @load="onImgLoad($event, art)" />
                        <!-- 悬停信息层 -->
                        <div class="overlay">
                            <p v-if="art.description" class="art-desc">{{ art.description }}</p>
                            <div class="art-meta">
                                <span class="art-title">{{ art.title }}</span>
                                <span class="art-time">{{ art.publishTime }}</span>
                            </div>
                        </div>
                    </div>
                </figure>
            </div>
        </div>

        <!-- 灯箱查看器 -->
        <Teleport to="body">
            <Transition name="viewer">
                <div v-if="viewer" class="viewer" @click="closeViewer" tabindex="0" ref="viewerEl">
                    <button class="viewer-close" aria-label="关闭" @click.stop="closeViewer">✕</button>

                    <button v-if="artworks.length > 1" class="viewer-nav prev" aria-label="上一张"
                        @click.stop="prev">‹</button>

                    <figure class="viewer-content" @click.stop>
                        <img :src="viewer.src" :alt="viewer.title" />
                        <figcaption class="viewer-info">
                            <div class="viewer-meta">
                                <span class="viewer-title">{{ viewer.title }}</span>
                                <span class="viewer-time">{{ viewer.publishTime }}</span>
                            </div>
                            <p v-if="viewer.description" class="viewer-desc">{{ viewer.description }}</p>
                        </figcaption>
                    </figure>

                    <button v-if="artworks.length > 1" class="viewer-nav next" aria-label="下一张"
                        @click.stop="next">›</button>
                </div>
            </Transition>
        </Teleport>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';

/* ---------- 类型 ---------- */
interface Artwork {
    src: string;
    title: string;
    description?: string;
    publishTime: string;
    aspect?: number;   // 宽 / 高，可选；不传则等图片加载后自动测量
}

/* ---------- 数据 ---------- */
const artworks = ref<Artwork[]>([
    {
        src: import.meta.env.BASE_URL + 'arts/oneswor_design_bingxizhu.jpg',
        title: 'OneSwot Design',
        description: "WHAT! Someone just told me they think she is kawaii instead of kowai! My anime art style = dark + cute?!",
        publishTime: '2026-11-06',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/oneswot_avatar_youlika_cry.png',
        title: '555...',
        description: "Don't worry, we just wanna see a cute JK sob a little... Not actually trying to make her sad.",
        publishTime: '2026-11-05',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/oneswot_avatar_anko_shock.jpeg',
        title: 'Anko: Nan Beng',
        description: "...",
        publishTime: '2026-11-03',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/oneswot_tachie_main_characters.jpg',
        title: 'OneSwot Design of 3 Main Characters',
        description: "You Li Ka (left), Klen Moee Li (middle), Shu Kwa Same (right). Err, why did I write this info just like a scientist?",
        publishTime: '2026-11-01',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_manga_adachifunou.jpg',
        title: '足立ちゃんの前で足が痺れて立たない二人',
        description: "A pun, hah? Manga from my original music: Try Catch Exception.",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_letsgo.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_miku_memory_leak.jpg',
        title: 'Miku Error: Out of Memory!',
        description: "Miku is All You Need!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_stack_overflow.jpg',
        title: 'Teto Error: Stack Overflow!',
        description: "Do you know? This stack of france pan is just like the logo of StackOverFlow, a website I often visited before we have ChatGPT!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_miku_onegai.jpg',
        title: 'お願い！たすけて！',
        description: "Sorry, but too many errors and warnings!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_deep_thinking.jpg',
        title: '深度思考中……',
        description: 'Do you know what dose " DO NOT EAT WHITE RICE" mean in Chinese?',
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_album_horizontal.jpg',
        title: 'Album of Try Catch Exception (Horizontal)',
        description: "I did not use this as the thumb of my video since I think it is too childlike",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_album_square.jpg',
        title: 'Album of Try Catch Exception',
        description: "I mean, the error icon at the center is a genius design! ね？",
        publishTime: '2026-07-27',
    },
]);

/* ---------- 列数与容器宽度 ---------- */
const galleryEl = ref<HTMLElement | null>(null);
const galleryWidth = ref<number>(0);
let resizeObserver: ResizeObserver | null = null;

const MIN_COL_WIDTH = 320;
const GAP = 20;

const colCount = computed<number>(() =>
    Math.max(1, Math.floor((galleryWidth.value + GAP) / (MIN_COL_WIDTH + GAP)))
);

onMounted(() => {
    if (!galleryEl.value) return;
    galleryWidth.value = galleryEl.value.clientWidth;
    resizeObserver = new ResizeObserver((entries: ResizeObserverEntry[]) => {
        let e = entries[0]
        if (e !== undefined) {
            galleryWidth.value = e.contentRect.width;
        }
    });
    resizeObserver.observe(galleryEl.value);

    window.addEventListener('keydown', onKeydown);
});

onBeforeUnmount(() => {
    resizeObserver?.disconnect();
    window.removeEventListener('keydown', onKeydown);
});

/* ---------- 宽高比 ---------- */
const aspects = ref<Map<string, number>>(new Map());

artworks.value.forEach((a: Artwork) => {
    if (a.aspect) aspects.value.set(a.src, a.aspect);
});

function onImgLoad(e: Event, art: Artwork): void {
    const img = e.target as HTMLImageElement;
    if (img.naturalWidth && img.naturalHeight) {
        aspects.value.set(art.src, img.naturalWidth / img.naturalHeight);
        aspects.value = new Map(aspects.value); // 触发响应式
    }
}

function estHeight(art: Artwork, colWidth: number): number {
    const aspect = aspects.value.get(art.src) ?? 0.75; // 默认竖图 3:4
    return colWidth / aspect;
}

/* ---------- 贪心填充 ---------- */
const columns = computed<Artwork[][]>(() => {
    const n = colCount.value;
    if (!n) return [];

    const colWidth = (galleryWidth.value - GAP * (n - 1)) / n;
    const cols: { items: Artwork[]; height: number }[] = Array.from(
        { length: n },
        () => ({ items: [], height: 0 })
    );

    for (const art of artworks.value) {
        let target = cols[0];
        if (target !== undefined) {
            for (const c of cols) {
                if (c.height < target.height) target = c;
            }
            target.items.push(art);
            target.height += estHeight(art, colWidth) + GAP;
        }

    }

    return cols.map((c) => c.items);
});

watch(artworks, (list: Artwork[]) => {
    aspects.value = new Map();
    list.forEach((a) => {
        if (a.aspect) aspects.value.set(a.src, a.aspect);
    });
});

/* ---------- 灯箱查看器 ---------- */
const viewer = ref<Artwork | null>(null);
const viewerEl = ref<HTMLElement | null>(null);

function openViewer(art: Artwork): void {
    viewer.value = art;
    document.body.style.overflow = 'hidden';
    viewerEl.value?.focus();
}

function closeViewer(): void {
    viewer.value = null;
    document.body.style.overflow = '';
}

function step(dir: -1 | 1): void {
    const i = artworks.value.findIndex((a) => a.src === viewer.value?.src);
    if (i < 0) return;
    const n = artworks.value.length;
    const art = artworks.value[(i + dir + n) % n];
    if (art !== undefined) {
        viewer.value = art
    }
}

const prev = (): void => step(-1);
const next = (): void => step(1);

function onKeydown(e: KeyboardEvent): void {
    if (!viewer.value) return;
    if (e.key === 'Escape') closeViewer();
    if (e.key === 'ArrowLeft') prev();
    if (e.key === 'ArrowRight') next();
}
</script>

<style scoped>
.art-page {
    width: 100%;
    padding: 20px;
    box-sizing: border-box;
}

.page-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 30px;
    color: var(--vt-c-text-light-1, #2c3e50);
    margin: 0 0 20px;
}

/* ---------- 瀑布流容器 ---------- */
.gallery {
    display: flex;
    gap: 20px;
    align-items: flex-start;
}

.gallery-col {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

/* ---------- 画作卡片 ---------- */
.art-card {
    margin: 0;
}

.img-wrapper {
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    background: var(--vt-c-white-mute, #f2f2f2);
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
    transition: box-shadow 0.4s;
    cursor: zoom-in;
}

.img-wrapper:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.img-wrapper img {
    display: block;
    width: 100%;
    max-width: 100%;
    height: auto;
    transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

.img-wrapper:hover img {
    transform: scale(1.06);
}

/* ---------- 悬停信息层 ---------- */
.overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding: 16px;
    background: linear-gradient(to top,
            rgba(0, 0, 0, 0.55) 0%,
            rgba(0, 0, 0, 0.15) 45%,
            transparent 70%);
    opacity: 0;
    transition: opacity 0.35s;
    pointer-events: none;
}

.img-wrapper:hover .overlay {
    opacity: 1;
}

.art-desc {
    font-family: 'TextRegular', sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: rgba(255, 255, 255, 0.92);
    margin: 0 0 10px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.art-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 12px;
}

.art-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 16px;
    color: #fff;
}

.art-time {
    font-family: 'CodeRegular', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.65);
    flex-shrink: 0;
}

/* ---------- 灯箱 ---------- */
.viewer {
    position: fixed;
    inset: 0;
    z-index: 100;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.85);
    outline: none;
}

.viewer-content {
    margin: 0;
    max-width: min(90vw, 1200px);
    max-height: 90vh;
    display: flex;
    flex-direction: column;
}

.viewer-content img {
    max-width: 100%;
    max-height: 78vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 12px 48px rgba(0, 0, 0, 0.5);
}

.viewer-info {
    padding: 14px 4px 0;
}

.viewer-meta {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 16px;
}

.viewer-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 18px;
    color: #fff;
}

.viewer-time {
    font-family: 'CodeRegular', monospace;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.65);
    flex-shrink: 0;
}

.viewer-desc {
    font-family: 'TextRegular', sans-serif;
    font-size: 14px;
    color: rgba(255, 255, 255, 0.85);
    margin: 6px 0 0;
}

/* 关闭与导航按钮 */
.viewer-close {
    position: absolute;
    top: 20px;
    right: 24px;
    font-size: 20px;
    color: rgba(255, 255, 255, 0.8);
    padding: 8px 12px;
    border-radius: 8px;
    transition: 0.3s;
}

.viewer-close:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.15);
}

.viewer-nav {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    font-size: 40px;
    line-height: 1;
    color: rgba(255, 255, 255, 0.7);
    padding: 16px 20px;
    border-radius: 8px;
    transition: 0.3s;
    user-select: none;
}

.viewer-nav:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.12);
}

.viewer-nav.prev {
    left: 20px;
}

.viewer-nav.next {
    right: 20px;
}

/* 淡入淡出 */
.viewer-enter-active,
.viewer-leave-active {
    transition: opacity 0.25s;
}

.viewer-enter-from,
.viewer-leave-to {
    opacity: 0;
}
</style>