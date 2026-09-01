<template>
    <div>Error</div>
    <div>
        <pre>?
            / \
            / \
            / /\|\ \
            \ O O /
            \ v /
            ---
        </pre>

        <pre>
            -
            | |
            | |
            |_|
            _
            (_)
        </pre>
    </div>

    <audio ref="panicAudio" src="/music/[重音テト]不知道在唱些什么的铁头酱.wav" loop></audio>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
const panicAudio = ref<any>(null)

onMounted(() => {
    if (panicAudio && panicAudio.value) {
        panicAudio.value.play().catch(() => {
            document.addEventListener('click', () => {
                panicAudio.value?.play()
            }, { once: true })
        })
    }

})
</script>