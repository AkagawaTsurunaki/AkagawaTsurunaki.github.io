/**
 * Reference:
 *  https://blog.woooo.tech/posts/marked_with_katex/
 * */
import katex, { type KatexOptions } from 'katex'
import 'katex/dist/katex.css'
import type { TokenizerAndRendererExtension } from 'marked'
import type { MarkedExtension } from 'marked'

export default function (options: KatexOptions = { throwOnError: false }): MarkedExtension {
  return {
    extensions: [inlineKatex(options), blockKatex(options)],
  }
}

function inlineKatex(options: KatexOptions): TokenizerAndRendererExtension {
  return {
    name: 'inlineKatex',
    level: 'inline',
    start(src: string) {
      return src.indexOf('$')
    },
    tokenizer(src: string, _tokens) {
      const match = src.match(/^\$+([^$\n]+?)\$+/)
      if (match) {
        if (match[1] === undefined) {
          return
        }
        return {
          type: 'inlineKatex',
          raw: match[0],
          text: match[1].trim(),
        }
      }
    },
    renderer(token) {
      /**
       * `options.displayMode` should be `false` for each render operation,
       * or it will be overwritten by the following statement in function `blockKatex`
       * ```
       * options.displayMode = true
       * ```
       * which affects the rendering result!
       *
       * Fixed by Github@AkagawaTsurunaki
       */
      options.displayMode = false
      return katex.renderToString(token.text, options)
    },
  }
}

function blockKatex(options: KatexOptions): TokenizerAndRendererExtension {
  return {
    name: 'blockKatex',
    level: 'block',
    start(src: string) {
      return src.indexOf('$$')
    },
    tokenizer(src: string, _tokens) {
      const match = src.match(/^\$\$+\n([^$]+?)\n\$\$/)
      if (match) {
        if (match[1] === undefined) {
          return
        }
        return {
          type: 'blockKatex',
          raw: match[0],
          text: match[1].trim(),
        }
      }
    },
    renderer(token) {
      options.displayMode = true
      return `<p>${katex.renderToString(token.text, options)}</p>`
    },
  }
}
