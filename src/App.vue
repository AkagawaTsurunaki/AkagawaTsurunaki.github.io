<script setup lang="ts">
import NavHeader from './components/NavHeader.vue'
</script>

<template>
  <NavHeader></NavHeader>

  <main>
    <div class="background">
      <RouterView />
    </div>
    <div class="copyright-container">©2025 AkagawaTsurunaki</div>
  </main>
</template>
<style scoped>
main {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.background {
  background-color: rgba(255, 255, 255, 1);
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  width: 90%;
}

.copyright-container {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
