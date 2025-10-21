<script setup lang="ts">
import { onMounted, ref } from 'vue';
import InfoIcon from "@/assets/icons/information.svg"
import WarnIcon from "@/assets/icons/warning.svg"
import ErrorIcon from "@/assets/icons/error.svg"
import { DialogLevel } from '@/scripts/dialog';

const level = ref<number>(DialogLevel.INFO)

function getIcon() {
    if (level.value === DialogLevel.INFO) {
        return InfoIcon;
    } else if (level.value === DialogLevel.WARN) {
        return WarnIcon;
    } else if (level.value === DialogLevel.ERROR) {
        return ErrorIcon;
    } else {
        return InfoIcon;
    }
}

const props = defineProps<{
    visible: boolean;
    level: number;
    title: string;
    content: string;
}>();

const visible = ref<boolean>(true)
function closeDialog() {
    visible.value = false;
}

const title = ref<string>("");
const content = ref<string>("");

onMounted(() => {
    visible.value = props.visible;
    level.value = props.level;
    title.value = props.title;
    content.value = props.content;
})

</script>

<template>
    <div v-if="visible" class="dialog-container">
        <div class="dialog-window">
            <div class="dialog-header">
                {{ title }}
            </div>
            <div class="dialog-body">
                <div class="dialog-icon-container">
                    <img class="dialog-icon" :src="getIcon()" alt="">
                </div>
                <div class="dialog-content">
                    <div>{{ content }}</div>
                    <div class="dialog-button-container">
                        <button @click="closeDialog">确认</button>
                        <button @click="closeDialog">取消</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
<style scoped>
.dialog-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 1000;
}

.dialog-header {
    background-color: rgb(255, 222, 95);
    border-radius: 5px 5px 0px 0px;
    padding: 5px 20px;
    font-family: "Header Font";
}

.dialog-window {
    background-color: white;
    /* padding: 20px; */
    border-radius: 5px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    width: 100%;
    max-width: 500px;
}

.dialog-body {
    display: inline-flex;
    width: 100%;
    padding: 30px;
    justify-content: space-between;
}

.dialog-content {
    display: flex;
    flex-direction: column;
    width: 80%;
    gap: 20px;
}

.dialog-icon-container {
    height: 100%;
}

.dialog-icon {
    width: 50px;
}

.dialog-button-container {
    display: inline-flex;
    /* flex-direction: row-reverse; */
    justify-content: right;
    gap: 10px;
}

.dialog-button-container button {
    border-radius: 20px;
    border: none;
    padding: 5px 20px;
}

.dialog-button-container button:hover {
    background-color: rgb(255, 222, 95);
    transition: 0.4s;
}
</style>