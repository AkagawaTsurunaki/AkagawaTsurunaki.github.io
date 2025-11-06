# AkagawaTsurunaki.github.io

How to deploy the website?

```bash
npm install
npm run prebuild
npm run build
npm run deploy
```

Prebuild command will load all blog metadata from `blogRegister.ts`, and generate `blogs.json` under `public/cache`.

Cache method can speed up loading blog items.

| Method     | Local Dev | Online Product |
| ---------- | --------- | -------------- |
| **cached** | 27.9 ms   | 787.6 ms       |
| **async**  | 33.5 ms   | 1579.7 ms      |
| **sync**   | 243.3 ms  | 18801.7 ms     |
