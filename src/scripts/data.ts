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

export class BlogItemVo {
    constructor(
        public id: number,
        public title: string,
        public tags: string[],
        public updatedTime: string,
        public preview: string,
        public filePath: string
    ) {}
}
