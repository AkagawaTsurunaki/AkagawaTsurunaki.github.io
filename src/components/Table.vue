<script setup lang="ts">
import { marked } from 'marked';
import { onMounted, ref, watch } from 'vue';

const props = defineProps<{ mdText: string }>()
const tableHtml = ref()
// watch(() => props.mdText, async (mdText) => {
//     tableHtml.value = await marked.parse(props.mdText)
// })

onMounted(async () => {
    tableHtml.value = await marked.parse(props.mdText)
})

</script>
<template>
    <div class="md-table" v-html="tableHtml"></div>
</template>
