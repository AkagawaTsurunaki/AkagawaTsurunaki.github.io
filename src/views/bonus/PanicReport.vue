<template>
    <div>Error</div>
    <audio ref="panicAudio" src="/music/[重音テト]不知道在唱些什么的铁头酱.wav" loop></audio>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const panicAudio = ref(null)

onMounted(() => {
    panicAudio.value.play().catch(() => {
        document.addEventListener('click', () => {
            panicAudio.value?.play()
        }, { once: true })
    })
})
</script>