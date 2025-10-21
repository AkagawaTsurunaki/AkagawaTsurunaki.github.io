export class HttpResponseBody<T> {
    code: number;
    message: string;
    data: T | null;

    constructor(code: number, message: string, data: T | null) {
        this.code = code;
        this.message = message;
        this.data = data;
    }

    public isSuccess(): boolean {
        return this.code === 0;
    }
}

export class TagDto {
    constructor(public name: string) {}
}

export class BlogItemDto {
    constructor(
        public title: string,
        public id: number,
        public tags: TagDto[],
        public updatedTime: number,
        public preview: string
    ) {}
}

export class BlogItemVo {
    constructor(
        public title: string,
        public id: number,
        public tags: string[],
        public updatedTime: string,
        public preview: string
    ) {}
}

export class BlogDetailDto {
    constructor(
        public id: number,
        public content: string
    ){}
}