// import DOMPurify from 'dompurify'
import { marked } from 'marked'

export async function renderMarkdown(text: string) {
  const katex = await import('@/scripts/katexRender')
  marked.use(katex.default({ strict: false }))
  // Use safe filter markdown text may cause katex error!
  // text = DOMPurify.sanitize(text);
  return await marked(text)
}
