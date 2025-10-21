<script setup lang="ts">
const props = defineProps({
    bgColor: {
        type: String,
        default: "#ffde5f"
    },
    tag: {
        type: String,
        default: "Tag"
    }
})
</script>
<template>
    <button class="tag-container" :style="{ backgroundColor: bgColor }">{{ tag }}</button>
</template>
<style scoped>
    .tag-container {
        border: none;
        border-radius: 20px;
        font-family: "Common Font";
        padding: 3px 20px;
    }
</style>