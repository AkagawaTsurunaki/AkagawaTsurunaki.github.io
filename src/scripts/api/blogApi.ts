import { plainToInstance } from 'class-transformer'
import { readFileJson } from './fileApi'
import { BlogItemDto } from '../data'
import { getBlogItemList } from '../build/blogRegister'
import { getMarkdownFileInfoByPath } from '../markdownUtil'

export async function getBlogItemListFromCache(): Promise<Array<BlogItemDto>> {
  const path = '/cache/blogs.json'
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

  const promises = blogRegList.map((blogReg, index) => {
    if (!blogReg?.mdFilePath) {
      return Promise.resolve(null) // 对于无效项，返回null
    }

    return getMarkdownFileInfoByPath(blogReg.mdFilePath)
      .then((info) => {
        if (!info) return null
        return new BlogItemDto(
          index,
          info?.title ? info?.title : '<无标题>',
          blogReg.tags,
          blogReg?.time,
          info?.preview,
          blogReg.mdFilePath,
        )
      })
      .catch((error) => {
        console.error(`Failed to get markdown info for ${blogReg.mdFilePath}:`, error)
        return null
      })
  })
  const results = await Promise.all(promises)
  const blogs = results.filter((blog) => blog !== null)

  return blogs
}
