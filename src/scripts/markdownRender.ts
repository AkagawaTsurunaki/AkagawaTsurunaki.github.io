import { marked } from 'marked'
import { h } from 'vue'
import type { VNode } from 'vue'
import CodeBlock from '@/components/CodeBlock.vue'

export async function renderMarkdown(src: string): Promise<VNode[]> {
  const katex = await import('@/scripts/katexRender')
  marked.use(katex.default({ strict: false }))

  // Inject custom components to AST of marked.
  const tokens = marked.lexer(src)
  const vNodes: VNode[] = []
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    if (token) {
      if (token.type === 'code') {
        // If code block
        vNodes.push(
          h(CodeBlock, {
            key: `code-${i}`,
            language: token.lang || '',
            code: token.text,
            rawHtml: null,
          }),
        )
      } else if (token.type === 'blockKatex') {
        const html = marked.parser([token])
        vNodes.push(
          h(CodeBlock, {
            key: `katex-${i}`,
            language: 'katex',
            code: token.text,
            rawHtml: html,
          }),
        )
      } else {
        // Wrap with v-node for other HTML content.
        const html = marked.parser([token])
        vNodes.push(
          h('div', {
            key: `html-${i}`,
            innerHTML: html,
          }),
        )
      }
    }
  }
  return vNodes
}
