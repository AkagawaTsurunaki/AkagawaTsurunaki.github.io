import { plainToInstance } from 'class-transformer'
import { readFileJson } from './fileApi'
import { BlogItemDto } from '../data'
import { getBlogItemList, getNoteItemList } from '../build/blogRegister'
import { getMarkdownFileInfoByPath } from '../markdownUtil'

export async function getBlogItemListFromCache(s: string): Promise<Array<BlogItemDto>> {
  const path = `/cache/${s}.json`
  const json = await readFileJson(path)
  if (json) {
    const blogs: Array<BlogItemDto> = plainToInstance(BlogItemDto, json)
    return blogs
  }
  return []
}

// This API should not use in product env
export async function getBlogItemListFromOnline(s: string): Promise<Array<BlogItemDto>> {
  let blogRegList = []
  if (s === 'blogs') {
    blogRegList = getBlogItemList()
  } else {
    blogRegList = getNoteItemList()
  }

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
