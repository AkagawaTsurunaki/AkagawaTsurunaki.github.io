<script setup lang="ts">
import { marked } from 'marked';
import { onMounted, ref } from 'vue';

const props = defineProps<{ mdText: string }>()
const tableHtml = ref()

onMounted(async () => {
    tableHtml.value = await marked.parse(props.mdText)
})

</script>
<template>
    <div class="md-table" v-html="tableHtml"></div>
</template>
