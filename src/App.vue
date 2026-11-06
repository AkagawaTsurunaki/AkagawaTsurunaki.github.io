<script setup lang="ts">
import NavHeader from './components/NavHeader.vue'
</script>

<template>
  <NavHeader></NavHeader>

  <main>
    <div class="background">
      <RouterView />
    </div>
    <div class="copyright-container">
      <p>©2025 AkagawaTsurunaki</p>
      <p>All Rights Reserved.</p>
    </div>
  </main>
</template>
<style scoped>
main {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-height: 100vh;
}

.background {
  background-color: rgba(255, 255, 255, 1);
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  width: 90%;
  flex: 1 0 auto;
}

.copyright-container {
  margin-top: 50px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  font-family: "TextRegular";
  font-size: 12px;
  color: rgb(91, 91, 91);
}
</style>
