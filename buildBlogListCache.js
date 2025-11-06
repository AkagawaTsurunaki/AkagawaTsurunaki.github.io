"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const blogRegister_1 = require("./blogRegister");
const data_1 = require("./data");
const markdownUtil_1 = require("./markdownUtil");
const fse = __importStar(require("fs-extra"));
const blogCachePath = 'public/cache/blogs.json';
console.log('??????????');
async function getBlogItemDtoList() {
    const blogItemList = (0, blogRegister_1.getBlogItemList)();
    const result = [];
    for (let index = 0; index < blogItemList.length; index++) {
        const blogItem = blogItemList[index];
        if (blogItem) {
            const fullPath = 'public/' + blogItem.mdFilePath;
            if (!(await fse.pathExists(fullPath))) {
                console.warn(`博客文件不存在: ${fullPath}`);
                continue;
            }
            const text = await fse.readFile(fullPath, 'utf-8');
            const mdInfo = (0, markdownUtil_1.getMarkdownFileInfo)(text);
            if (!mdInfo)
                continue;
            if (!mdInfo.title)
                continue;
            const tags = blogItem?.tags || [];
            const updateTime = blogItem?.time || '';
            const dto = new data_1.BlogItemDto(index, mdInfo?.title, tags, updateTime, mdInfo?.preview, blogItem.mdFilePath);
            result.push(dto);
        }
    }
    return result;
}
async function saveCache() {
    try {
        const blogs = await getBlogItemDtoList();
        const json = JSON.stringify(blogs);
        console.log(json);
    }
    catch (error) {
        console.error('生成缓存失败:', error);
    }
}
saveCache();
