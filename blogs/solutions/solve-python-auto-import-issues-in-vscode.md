# 解决VS Code中Python无法自动导入某些模块、符号的问题

我用 VS Code 写了很长时间的代码，尤其最近研究深度学习，需要用 SSH 连接到远程服务器进行开发。

编写 Python 代码的过程中，我发现某些包、模块或者符号使用快捷键 Ctrl+. 始终无法自动导入，但是有些却可以。这与 PyCharm 的使用体验截然不同，PyCharm 几乎可以在编写的过程中就自动补全一些 import 语句，我一度怀疑这个所谓的“宇宙最强 IDE”是否有些垃圾……

后来我在 Github 上找到的这个关于 Pylance 插件的一个 Issue，才算是把根本原因搞明白。

[Pylance - searchImport not finding modules #5843](https://github.com/microsoft/pylance-release/issues/5843)

**原来是 Pylance 的默认包索引深度为 1，导致它根本就没有往更深层次的包里看。**

那么解决方案就是把这个包索引深度增大即可。

### **具体做法**

首先，我们要找到设置文件。先找到 VS Code 左下角的设置图标，然后在上面输入 python.analysis.packageIndexDepths，随后回车会切换到一个配置界面，在下面点击 Edit in setting.json，就可以打开设置文件界面了。 

![img](/images/vscode-setting-page.png)


如何打开设置文件

打开设置文件界面后，在 python.analysis.packageIndexDepths 数组那里添加一个 JSON 对象：

```json
{ 
     "name": "", 
     "depth": 100,  
     "includeAllSymbols": true 
}
```

![img](/images/vscode-settings-pacakge-index-depths.png)

添加图片注释，不超过 140 字（可选）

这些配置项的含义是，不指定包名即针对所有的包，索引深度都最大为 100，且包含所有的符号。

需要注意的是，这种操作势必会**增加VS Code运行时的负载，也可能会降低运行性能**，所以可以根据需要适当调整。

还有，别忘了把**远程服务器上的配置也同样改了**。在修改后，**务必重启 VS Code**，这样配置才能生效！

我还试着把 depth 改成 -1，但这并不起作用，大概是怕超过索引函数的递归上限吧，所以大家还是指定一个稍大点的正整数吧。

这是配置生效后的效果图，原先是搜索不到的，这就说明我们成功了：

![img](/images/vscode-import-deep-package.png)

成功搜索到更深层次的包！

关于 python.analysis.packageIndexDepths 配置项的更多信息，可以查看 Pylance 的官方文档：

[pylance-release/docs/settings/python_analysis_packageIndexDepths.md at main · microsoft/pylance-release](https://github.com/microsoft/pylance-release/blob/main/docs/settings/python_analysis_packageIndexDepths.md)