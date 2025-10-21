import Dialog from "@/components/Dialog.vue";
import { createApp, h } from "vue";

export enum DialogLevel {
  INFO,
  WARN,
  ERROR,
}

export function openDialog(level: DialogLevel, title: string, content: string) {
  const div = document.createElement("div");
  document.body.appendChild(div);
  const vnode = h(Dialog, {
    visible: true,
    level: level,
    title: title,
    content: content,
  });

  const app = createApp({
    setup() {
      return () => vnode;
    },
  });

  app.mount(div);
}
