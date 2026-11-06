<template>
    <aside class="music-list">
        <h2 class="list-title">歌单</h2>
        <ul class="list">
            <li v-for="song in songs" :key="song.id" :class="{ active: song.id === selectedId }" class="song-item"
                @click="$emit('select', song)">
                <div class="song-info">
                    <span class="song-title">{{ song.title }}</span>
                    <span class="song-artist">{{ song.artist }}</span>
                </div>
                <span v-if="song.id === selectedId" class="playing-dot" aria-label="当前播放"></span>
            </li>
        </ul>
    </aside>
</template>

<script setup>
defineProps({
    songs: { type: Array, default: () => [] }, // [{ id, title, artist }]
    selectedId: { type: [Number, String], default: null },
});

defineEmits(['select']);
</script>

<style scoped>
.music-list {
    display: flex;
    flex-direction: column;
    height: 80vh;
    background: var(--color-background-soft, #f8f8f8);
    border: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
    border-radius: 12px;
    overflow: hidden;
}

.list-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 20px;
    color: var(--vt-c-text-light-1, #2c3e50);
    padding: 16px 18px 12px;
    margin: 0;
    border-bottom: 1px solid var(--color-border, rgba(60, 60, 60, 0.12));
}

.list {
    list-style: none;
    margin: 0;
    padding: 8px;
    overflow-y: auto;
    flex: 1;
}

.list::-webkit-scrollbar {
    width: 6px;
}

.list::-webkit-scrollbar-thumb {
    background: var(--vt-c-divider-light-1);
    border-radius: 3px;
}

.song-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.song-item:hover {
    background: var(--vt-c-white-mute, #f2f2f2);
}

.song-item.active {
    background: var(--akt-c-yellow, rgb(255, 222, 95));
}

.song-info {
    display: flex;
    flex-direction: column;
    min-width: 0;
}

.song-title {
    font-family: 'TextBold', sans-serif;
    font-weight: 900;
    font-size: 15px;
    color: var(--vt-c-text-light-1, #2c3e50);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.song-artist {
    font-family: 'TextRegular', sans-serif;
    font-size: 12px;
    color: var(--vt-c-text-light-2, rgba(60, 60, 60, 0.66));
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* 当前播放的小圆点，黄色底色上的深色点 */
.playing-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--vt-c-text-light-1, #2c3e50);
    flex-shrink: 0;
}
</style>