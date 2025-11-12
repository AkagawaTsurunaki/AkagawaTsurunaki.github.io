import { marked, type Token, type Tokens } from 'marked'
import { readFileText } from './api/fileApi'
import { HeaderTreeNode } from './data'
import 'marked'

export async function getMarkdownFileInfoByPath(
  publicPath: string,
  maxLength: number = 200,
): Promise<{ title: string | null; preview: string } | null> {
  const text = await readFileText(publicPath)
  if (text) {
    return getMarkdownFileInfo(text, maxLength)
  }
  return null
}

export function getMarkdownFileInfo(text: string, maxLength: number = 200) {
  try {
    // If we get the title of such a markdown file, then return the other lines below the title.
    // Else we just return all preview content if we can not find the title.
    const index = getIndexOfFirstH1(text)
    if (index < 0) {
      return { title: null, preview: text.slice(0, Math.max(0, maxLength)) + '...' }
    }

    const preview = text.slice(index, index + Math.max(0, maxLength))
    const indexOfFirstBr = getIndexOfFirstBr(preview)
    if (indexOfFirstBr > 0) {
      return {
        title: preview.slice(2, indexOfFirstBr + 1),
        preview:
          preview.slice(indexOfFirstBr + 1, indexOfFirstBr + 1 + Math.max(0, preview.length)) +
          '...',
      }
    }
    // Only have title but no more content
    return {
      title: null,
      preview: preview.slice(2, 2 + Math.max(0, preview.length)) + '...',
    }
  } catch (err) {
    console.error(err)
    return null
  }
}

function getIndexOfFirstH1(text: string) {
  for (let index = 0; index < text.length - 1; index++) {
    if (text[index] == '#' && text[index + 1] == ' ') {
      return index
    }
  }
  return -1
}

function getIndexOfFirstBr(str: string) {
  for (let index = 0; index < str.length; index++) {
    if (str[index] == '\n') {
      return index
    }
  }
  return -1
}

export function parseMarkdownToc(markdownContent: string): HeaderTreeNode {
  const tokens = marked.lexer(markdownContent)
  const headingTokens = tokens.filter((token) => token.type === 'heading') as Tokens.Heading[]
  console.log(headingTokens)
  const root = new HeaderTreeNode('', 0, '', [])
  buildTree(headingTokens, root)
  console.log(root)
  return root
}

export function buildTree(tokens: Tokens.Heading[], root: HeaderTreeNode): void {
  // 栈顶永远是“当前节点”的父节点
  const stack: HeaderTreeNode[] = [root]

  tokens.forEach((tok) => {
    const level = tok.depth
    const text = tok.text
    // 生成 id：github 风格锚点，空格变 -，去掉特殊符号
    const id = text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s-]/gu, '') // 去掉标点
      .trim()
      .replace(/\s+/g, '-')

    // 1. 找到正确的父：栈中最后一个 level < 当前 level
    while (stack.length > 1 && stack[stack.length - 1].level >= level) {
      stack.pop()
    }

    // 2. 创建新节点
    const node = new HeaderTreeNode(id, level, text, [])

    // 3. 挂到父节点
    stack[stack.length - 1].children.push(node)

    // 4. 自己入栈，成为下一个节点的候选父
    stack.push(node)
  })
}
