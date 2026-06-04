<<template>
    <div class="lazy-image" ref="lazyContainer" :class="{ loading: isLoading }">
        <ImageSkeleton v-if="isLoading" />
        <img v-else :src="imageUrl" :alt="altText" :width="width" :height="height" @load="onImageLoad" loading="lazy" />
    </div>
</template>

    <style>
    .lazy-image {
        display: block;
        /* 原来是 inline，改为 block 才能控制尺寸 */
        width: 100%;
        min-height: 100px;
        /* 与骨架屏高度一致，避免塌陷 */
        position: relative;
    }

    .lazy-image img {
        display: block;
        max-width: 100%;
        height: auto;
    }
</style>

    <script setup lang="ts">
    import ImageSkeleton from '@/components/ImageSkeleton.vue'
    import { onMounted, onBeforeUnmount, ref } from 'vue';

    const props = defineProps<{
        imageUrl: string,
        altText: string,
        width: number | undefined,
        height: number | undefined
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

    function onImageLoad() {
        // 图片加载完成后，如果容器高度变化，这里可以做一些平滑处理
    }

    onBeforeUnmount(() => {
        if (observer.value) {
            observer.value.disconnect();
        }
    })
</script>