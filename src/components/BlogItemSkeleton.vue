<template>
    <div class="blog-item-skeleton">
        <div class="skeleton-content">
            <div class="skeleton-line skeleton-title"></div>
            <div class="skeleton-line skeleton-meta"></div>
            <div class="skeleton-line skeleton-text"></div>
            <div class="skeleton-line skeleton-text"></div>
            <div class="skeleton-line skeleton-text-short"></div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'BlogItemSkeleton'
}
</script>

<style scoped>
.blog-item-skeleton {
    width: 100%;
    padding: 16px;
    background: #f8f9fa;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    margin-bottom: 16px;
}

.skeleton-content {
    position: flex;
}

.skeleton-line {
    background-color: #e0e0e0;
    border-radius: 4px;
    margin-bottom: 12px;
    height: 16px;
    position: relative;
    overflow: hidden;
}

.skeleton-title {
    height: 20px;
    width: 70%;
    margin-bottom: 16px;
}

.skeleton-meta {
    width: 40%;
    height: 14px;
    margin-bottom: 20px;
}

.skeleton-text {
    width: 100%;
}

.skeleton-text-short {
    width: 85%;
}

.skeleton-line::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg,
            transparent,
            rgba(255, 255, 255, 0.4) 20%,
            rgba(255, 255, 255, 0.2) 50%,
            rgba(255, 255, 255, 0.4) 80%,
            transparent);
    animation: shimmer 2s infinite;
}

@keyframes shimmer {
    0% {
        transform: translateX(-100%);
    }

    100% {
        transform: translateX(100%);
    }
}
</style>