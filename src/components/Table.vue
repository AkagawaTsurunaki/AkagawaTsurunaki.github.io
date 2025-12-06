<script setup lang="ts">
import { markedInstance } from '@/scripts/render/markdownRender';
import { onMounted, ref } from 'vue';

const props = defineProps<{ mdText: string }>()
const tableHtml = ref()

onMounted(async () => {
    tableHtml.value = await markedInstance.parse(props.mdText)
})

</script>
<template>
    <div class="md-table" v-html="tableHtml"></div>
</template>
