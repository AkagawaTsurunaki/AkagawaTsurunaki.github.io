<template>
    <div>
        <button>111</button>
        <span>{{ language }}</span>
    </div>
    <!-- If you add breakline after code tag, will cause code block display another line below, but why? -->
    <pre class="code-block"><code>{{ code }}</code></pre>
</template>

<script setup lang="ts">
defineProps<{ language: string; code: string }>();
</script>

<style scoped>
.code-block {
    background-color: #f6f8fa;
    color: #2c3e50;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: "Consolas";
    font-size: 85%;
}
</style>