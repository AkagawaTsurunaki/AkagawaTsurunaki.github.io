<script setup lang="ts">
import NavHeader from './components/NavHeader.vue'
</script>

<template>
  <NavHeader></NavHeader>

  <body class="background">
    <main>
      <div class="white-bg">
        <RouterView />
      </div>
      <div class="copyright-container">©2025 AkagawaTsurunaki</div>
    </main>
  </body>
</template>
<style scoped>
@font-face {
  font-family: 'Common Font';
  src: url('src/assets/fonts/HarmonyOS_Sans_Medium.ttf') format('truetype');
  font-weight: 400;
  /* Black 字重 */
}

.background {
  background-position: center center;
  background-repeat: no-repeat;
  background-attachment: fixed;
  background-size: cover;
  background-color: #ffffff;
}

.white-bg {
  background-color: rgba(255, 255, 255, 1);
  width: auto;
}

.copyright-container {
  display: flex;
  justify-content: center;
  align-items: center;
}

main {
  display: flex;
  flex-direction: column;
}
</style>
