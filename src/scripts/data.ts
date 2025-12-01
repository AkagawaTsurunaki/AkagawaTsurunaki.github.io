import { type VNode } from 'vue'
export class HttpResponseBody<T> {
  code: number
  message: string
  data: T | null

  constructor(code: number, message: string, data: T | null) {
    this.code = code
    this.message = message
    this.data = data
  }

  public isSuccess(): boolean {
    return this.code === 0
  }
}

export class BlogItemDto {
  constructor(
    public id: number,
    public title: string,
    public tags: string[],
    public updatedTime: string,
    public preview: string,
    public filePath: string,
  ) {}
}

export class Header {
  constructor(
    public id: string, // 对应正文的 h1/h2/h3… id，用于锚点跳转
    public level: number, // 1~6 方便做缩进
    public text: string, // 标题文字
  ) {}
}

export class MarkdownDto {
  constructor(
    public nodes: VNode[],
    public toc: Array<Header>,
  ) {}
}
