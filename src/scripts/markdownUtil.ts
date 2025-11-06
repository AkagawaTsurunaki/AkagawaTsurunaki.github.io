import { readFileText } from './api/fileApi'

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
