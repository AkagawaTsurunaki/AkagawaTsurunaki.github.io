import { getBlogItemList } from './blogRegister.ts'
import { BlogItemDto } from '../data.ts'
import { getMarkdownFileInfo } from '../markdownUtil.ts'
import fse from 'fs-extra'

const blogCachePath = 'public/cache/blogs.json'

async function getBlogItemDtoList(): Promise<Array<BlogItemDto>> {
  const blogItemList = getBlogItemList()
  const result: Array<BlogItemDto> = []

  for (let index = 0; index < blogItemList.length; index++) {
    const blogItem = blogItemList[index]
    if (blogItem) {
      const fullPath = 'public/' + blogItem.mdFilePath
      if (!(await fse.pathExists(fullPath))) {
        console.warn(`博客文件不存在: ${fullPath}`)
        continue
      }
      const text = await fse.readFile(fullPath, 'utf-8')
      const mdInfo = getMarkdownFileInfo(text)
      if (!mdInfo) continue
      if (!mdInfo.title) continue
      const tags = blogItem?.tags || []
      const updateTime = blogItem?.time || ''
      const dto = new BlogItemDto(
        index,
        mdInfo?.title,
        tags,
        updateTime,
        mdInfo?.preview,
        blogItem.mdFilePath,
      )
      result.push(dto)
    }
  }

  return result
}

async function saveCache() {
  try {
    console.log('Building blogs cache...')
    const blogs = await getBlogItemDtoList()
    fse.writeJson(blogCachePath, blogs, { spaces: 2 })
    console.log('Blogs cache saved.')
  } catch (error) {
    console.error('Error to build blogs cache: \n', error)
  }
}

saveCache()
