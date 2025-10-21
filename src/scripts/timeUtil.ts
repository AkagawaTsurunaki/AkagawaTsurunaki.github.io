export function getTimeString(timestamp: number | null = null) {
  const now = timestamp === null ? new Date() : new Date(timestamp);
  const formattedDate = now.toLocaleDateString("zh-CN");
  const time = now.toLocaleTimeString("zh-CN");
  return formattedDate + " " + time;
}
