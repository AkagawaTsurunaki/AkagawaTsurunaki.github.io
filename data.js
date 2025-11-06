"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BlogItemDto = exports.BlogItemVo = exports.HttpResponseBody = void 0;
class HttpResponseBody {
    constructor(code, message, data) {
        this.code = code;
        this.message = message;
        this.data = data;
    }
    isSuccess() {
        return this.code === 0;
    }
}
exports.HttpResponseBody = HttpResponseBody;
class BlogItemVo {
    constructor(id, title, tags, updatedTime, preview, filePath) {
        this.id = id;
        this.title = title;
        this.tags = tags;
        this.updatedTime = updatedTime;
        this.preview = preview;
        this.filePath = filePath;
    }
}
exports.BlogItemVo = BlogItemVo;
class BlogItemDto {
    constructor(id, title, tags, updatedTime, preview, filePath) {
        this.id = id;
        this.title = title;
        this.tags = tags;
        this.updatedTime = updatedTime;
        this.preview = preview;
        this.filePath = filePath;
    }
}
exports.BlogItemDto = BlogItemDto;
