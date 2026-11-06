<template>
    <div class="art-page">
        <h1 class="page-title">画作</h1>

        <div class="gallery" ref="galleryEl">
            <div v-for="(col, ci) in columns" :key="ci" class="gallery-col">
                <figure v-for="art in col" :key="art.src" class="art-card" @click="$emit('select', art)">
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
    </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue';

const artworks = ref([
    {
        src: import.meta.env.BASE_URL + 'arts/oneswot_tachie_main_characters.jpg',
        title: 'OneSwot Design',
        description: "The 3 main characters of my original RPG game",
        publishTime: '2026-11-07',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_manga_adachifunou.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
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
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_stack_overflow.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_miku_onegai.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_teto_deep_thinking.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_album_horizontal.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },
    {
        src: import.meta.env.BASE_URL + 'arts/tce_album_square.jpg',
        title: '立ちました！フランスパン！',
        description: "Miku: Let's go Teto-chan! Teto: Wait! Wait!",
        publishTime: '2026-07-27',
    },

]);

defineEmits(['select']);

/* ---------- 列数与容器宽度 ---------- */
const galleryEl = ref(null);
const galleryWidth = ref(0);
let resizeObserver = null;

const MIN_COL_WIDTH = 320;   // 每列最小宽度，按这个决定列数
const GAP = 20;              // 与 CSS 里的 gap 保持一致

const colCount = computed(() =>
    Math.max(1, Math.floor((galleryWidth.value + GAP) / (MIN_COL_WIDTH + GAP)))
);

onMounted(() => {
    galleryWidth.value = galleryEl.value?.clientWidth ?? 0;
    resizeObserver = new ResizeObserver(entries => {
        galleryWidth.value = entries[0].contentRect.width;
    });
    resizeObserver.observe(galleryEl.value);
});

onBeforeUnmount(() => resizeObserver?.disconnect());

/* ---------- 宽高比：先从 props 拿，图片加载后校准 ---------- */
const aspects = ref(new Map()); // src -> width / height

artworks.value.forEach(a => {
    if (a.aspect) aspects.value.set(a.src, a.aspect);
});

function onImgLoad(e, art) {
    const img = e.target;
    if (img.naturalWidth && img.naturalHeight) {
        aspects.value.set(art.src, img.naturalWidth / img.naturalHeight);
        aspects.value = new Map(aspects.value); // 触发响应式
    }
}

function estHeight(art, colWidth) {
    const aspect = aspects.value.get(art.src) ?? 0.75; // 默认竖图 3:4
    return colWidth / aspect;
}

/* ---------- 贪心填充：每幅放入当前累计高度最矮的列 ---------- */
const columns = computed(() => {
    const n = colCount.value;
    if (!n) return [];

    const colWidth = (galleryWidth.value - GAP * (n - 1)) / n;
    const cols = Array.from({ length: n }, () => ({ items: [], height: 0 }));

    for (const art of artworks.value) {
        let target = cols[0];
        for (const c of cols) {
            if (c.height < target.height) target = c;
        }
        target.items.push(art);
        target.height += estHeight(art, colWidth) + GAP;
    }

    return cols.map(c => c.items);
});

//  artworks 列表本身变化时也重排
watch(artworks, list => {   // ✅ 响应式数据是本地 ref 时，直接传即可
    aspects.value = new Map();
    list.forEach(a => {
        if (a.aspect) aspects.value.set(a.src, a.aspect);
    });
});  // 不需要 deep: true，ref 整体替换天然可侦听
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
}

.img-wrapper:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.img-wrapper img {
    display: block;
    width: 100%;
    max-width: 100%;
    /* 新增：推翻 base.css 的全局 80% */
    height: auto;
    transition: transform 0.6s cubic-bezier(0.22, 1, 0.36, 1);
}

.img-wrapper:hover img {
    transform: scale(1.06);
    /* 悬停放大 */
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
    /* 不挡图片的其它交互 */
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
    /* 描述最多三行，超出省略 */
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

/* 淡白色的发布时间 */
.art-time {
    font-family: 'CodeRegular', monospace;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.65);
    flex-shrink: 0;
}
</style>