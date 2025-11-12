import { marked } from 'marked'
import { h } from 'vue'
import type { VNode } from 'vue'
import CodeBlock from '@/components/CodeBlock.vue'
import Image from '../../components/Image.vue'
import { parse } from 'node-html-parser'
import MermaidBlock from '@/components/MermaidBlock.vue'
import { parseMarkdownToc, slugify } from '../markdownUtil'
import { MarkdownDto } from '../data'

export async function renderMarkdown(
  src: string,
  skip: undefined | Array<string> = undefined,
): Promise<MarkdownDto> {
  const katex = await import('@/scripts/render/katexRender')
  marked.use(katex.default({ strict: false }))

  const renderer = new marked.Renderer()
  renderer.heading = ({ text, depth }) => {
    const id = slugify(text)
    return `<h${depth} id="${id}">${text}</h${depth}>`
  }
  marked.use({ renderer })

  // Inject custom components to AST of marked.
  const tokens = marked.lexer(src)
  const headers = parseMarkdownToc(tokens)

  const vNodes: VNode[] = []
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    if (token) {
      if (token.type === 'code') {
        if (token.lang === 'mermaid') {
          vNodes.push(
            h(MermaidBlock, {
              key: `code-${i}`,
              language: token.lang || '',
              code: token.text,
              rawHtml: null,
            }),
          )
        } else {
          // If code block
          vNodes.push(
            h(CodeBlock, {
              key: `code-${i}`,
              language: token.lang || '',
              code: token.text,
              rawHtml: null,
            }),
          )
        }
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
      } else if (token.type === 'image') {
        if (skip?.includes('image')) continue
        vNodes.push(
          h(Image, {
            imageUrl: token.href || '',
            altText: token.text || '',
            width: undefined,
            height: undefined,
            key: `image-${i}`,
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
              key: `image-${i}`,
            }),
          )
        } else {
          vNodes.push(
            h('div', {
              key: `html-${i}`,
              innerHTML: html,
            }),
          )
        }
      }
    }
  }
  return new MarkdownDto(vNodes, headers)
}
