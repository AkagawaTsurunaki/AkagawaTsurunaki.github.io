<template>
    <div class="lazy-image" ref="lazyContainer" :class="{ loading: isLoading }">
        <ImageSkeleton v-if="isLoading" />
        <img v-else :src="imageUrl" :alt="altText" :width="width" :height="height" />
    </div>
</template>

<script setup lang="ts">
import ImageSkeleton from '@/components/ImageSkeleton.vue'
import { onMounted, onBeforeUnmount, ref } from 'vue';

const props = defineProps<{
    imageUrl: string;
    altText: string;
    width: number;
    height: number;
}>();

const isLoading = ref(true)
const observer = ref<IntersectionObserver>();
const lazyContainer = ref<HTMLElement | null>(null);

onMounted(() => {
    observer.value = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                loadImage();
                if (observer.value)
                    observer.value.disconnect();
            }
        });
    });

    if (lazyContainer.value) {
        observer.value.observe(lazyContainer.value);
    }
})

function loadImage() {
    const img = new Image();
    img.onload = () => {
        isLoading.value = false;
    };
    img.src = props.imageUrl;
}

onBeforeUnmount(() => {
    if (observer.value) {
        observer.value.disconnect();
    }
})
</script>
