import { marked } from 'marked'
import { h } from 'vue'
import type { VNode } from 'vue'
import CodeBlock from '@/components/CodeBlock.vue'
import Image from '../../components/Image.vue'
import { parse } from 'node-html-parser'
import MermaidBlock from '@/components/MermaidBlock.vue'
import { parseMarkdownToc, slugify } from '../markdownUtil'
import { MarkdownDto } from '../data'
import Table from '@/components/Table.vue'
import hljs from 'highlight.js'

const katex = await import('@/scripts/render/katexRender')
marked.use(katex.default({ strict: false }))

export async function renderMarkdown(
  src: string,
  skip: undefined | Array<string> = undefined,
): Promise<MarkdownDto> {
  // Inject custom components to AST of marked.
  const tokens = marked.lexer(src)
  const headers = parseMarkdownToc(tokens)

  const vNodes: VNode[] = []
  let headerNum = 1
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    if (token) {
      if (token.type === 'code') {
        if (token.lang === 'mermaid') {
          vNodes.push(
            h(MermaidBlock, {
              id: `code-${i}`,
              language: token.lang || '',
              code: token.text,
              rawHtml: null,
            }),
          )
        } else {
          // If code block
          vNodes.push(
            h(CodeBlock, {
              id: `code-${i}`,
              language: token.lang || '',
              code: token.text,
              rawHtml: highlightCode(token.text, token.lang),
            }),
          )
        }
      } else if (token.type === 'blockKatex') {
        const html = marked.parser([token])
        vNodes.push(
          h(CodeBlock, {
            id: `katex-${i}`,
            language: 'katex',
            code: token.text,
            rawHtml: html,
          }),
        )
      } else if (token.type === 'image') {
        if (skip?.includes('image')) continue
        vNodes.push(
          h(Image, {
            imageUrl: token.href || '',
            altText: token.text || '',
            width: undefined,
            height: undefined,
            id: `image-${i}`,
          }),
        )
      } else if (token.type === 'table') {
        vNodes.push(h(Table, { mdText: token.raw }))
      } else if (token.type === 'heading') {
        if (skip?.includes('heading')) continue
        const id = `${slugify(token.text)} ${headerNum++}`
        let html = await marked.parse(token.raw)
        html = html.substring(4, html.length - 6)
        vNodes.push(
          h(`h${token.depth}`, {
            id: id,
            innerHTML: html,
            class: 'md-heading',
          }),
        )
      } else {
        // Wrap with v-node for other HTML content.
        const html = marked.parser([token])

        // Parse image
        if (html.includes('img') && html.includes('src=')) {
          if (skip?.includes('image')) continue
          const img = parse(html).querySelector('img')
          vNodes.push(
            h(Image, {
              imageUrl: img?.getAttribute('src') || '',
              altText: img?.getAttribute('alt') || '',
              width: Number(img?.getAttribute('width')) || undefined,
              height: Number(img?.getAttribute('height')) || undefined,
              id: `image-${i}`,
            }),
          )
        } else {
          vNodes.push(
            h('div', {
              id: `html-${i}`,
              innerHTML: html,
            }),
          )
        }
      }
    }
  }
  return new MarkdownDto(vNodes, headers)
}

export function highlightCode(code: string, lang?: string) {
  const language = hljs.getLanguage(lang || '') ? lang : 'plaintext'
  return hljs.highlight(code, { language: language, ignoreIllegals: true }).value
}
