import DOMPurify from 'dompurify';
import { marked } from 'marked';

export async function renderMarkdown(text: string) {
  const katex = await import("@/scripts/katexRender")
  marked.use(katex.default({ strict: false }))
  // text = DOMPurify.sanitize(text);
  return await marked(text);
}