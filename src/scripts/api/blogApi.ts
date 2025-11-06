import { plainToInstance } from 'class-transformer'
import { readFileJson } from './fileApi'
import { BlogItemDto } from '../data'
import { getBlogItemList } from '../blogRegister'
import { getMarkdownFileInfoByPath } from '../markdownUtil'

export async function getBlogItemListFromCache(): Promise<Array<BlogItemDto>> {
  const path = 'public/cache/blogs.json'
  const json = await readFileJson(path)
  if (json) {
    const blogs: Array<BlogItemDto> = plainToInstance(BlogItemDto, json)
    return blogs
  }
  return []
}

// This API should not use in product env
export async function getBlogItemListFromOnline(): Promise<Array<BlogItemDto>> {
  const blogRegList = getBlogItemList()
  const blogs = []
  for (let index = 0; index < blogRegList.length; index++) {
    const blogReg = blogRegList[index]
    if (blogReg?.mdFilePath) {
      const info = await getMarkdownFileInfoByPath(blogReg?.mdFilePath)
      if (info) {
        const blog = new BlogItemDto(
          index,
          info?.title ? info?.title : '<无标题>',
          blogReg.tags,
          blogReg?.time,
          info?.preview,
          blogReg.mdFilePath,
        )
        blogs.push(blog)
      }
    }
  }
  return blogs
}
