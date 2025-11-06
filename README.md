# AkagawaTsurunaki.github.io

How to deploy the website?

```bash
npm install
npm run prebuild
npm run build
npm run deploy
```

Prebuild command will load all blog metadata from `blogRegister.ts`, and generate `blogs.json` under `public/cache`.
