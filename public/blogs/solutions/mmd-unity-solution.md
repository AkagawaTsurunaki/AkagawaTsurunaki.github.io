# Unity 6 如何正常导入 MMD 模型？Material/Shader 异常

最近需要用 Unity 6 开发一些 VR 项目，结果导入 MMD 模型后居然变成粉色唐人了（Material/Shader 异常）？！而且网上也没找到针对高版本 Unity 怎么解决的方法！好在本人经过一番摸索终于把它搞明白了！

<img src="/images/anon-watching-phone.png" width="200">

## 导入 MMD4Mecanim

在 Unity 中支持 MMD，首先我们需要下载 [MMD4Mecanim](https://stereoarts.jp/)（我这里的版本是 `MMD4Mecanim_Beta_20200105`），然后将下载好的压缩包解压后，拖动 `MMD4Mecanim.unitypackage` 到 Unity 项目中导入。

需要注意，带 Standard 字样的那个包功能更少，貌似只包含了 Shader 文件，不适合开发使用。

如果你解压后有乱码文件，在安装了 7zip 的环境下可以运行：

```shell
7z x MMD4Mecanim_Beta_20200105.zip -mcp=932
```

可能还会遇到提醒需不需要更新 precompiled assemblies（如下图），我试了更新和不更新貌似没有区别。

![mmd4macanim-import-content-request](/images/mmd4macanim-import-content-request.png)

一般来说，导入 MMD4Mecanim 成功后，你就可以在 `Assets\MMD4Mecanim` 路径下看到它了。

## 导入 MMD 模型

首先，从网上下载一些 MMD 模型，解压后里面可能会包含一个 PMX 扩展名的模型。我这里是从 Booth 上随便下载的一个免费的二次元模型 Hoshifuri_Iku，里面有一个名为 `MMD Hoshifuri Iku v1.0.pmx` 文件就是 MMD 的模型文件。文件夹的结构类似下面这样：

```
├─Hoshifuri_Iku_MMD_v1.0_by_Itsuki_Fukurou
   │  Hoshifuri Iku - special blendshapes morphes.txt
   │  MMD Hoshifuri Iku v1.0.pmx  <--- 这个就是 PMX 文件，也就是人物模型文件
   │
   └─textures                     <--- 这个文件夹下面是材质的图片
           hsfrIku_blouse tex.png
           hsfrIku_body lower tex.png
           ...
           toon-skirt.png
```

然后，使用 Blender 打开该文件，这一步实际上是为了检查你的模型文件本身有没有问题。Blender 导入 MMD 的教程也是随处可搜。我下载的模型大概长这样：

![mmd-in-blender](/images/mmd-in-blender.png)

嗯，在 Blender 里看起来是正常的。

接着，将模型的整个文件夹 `Hoshifuri_Iku_MMD_v1.0_by_Itsuki_Fukurou` 都拖进 Unity 去，会看见有一个叫 `MMD Hoshifuri Iku v1.0.MMD4Mecanim` 的文件在这个文件夹下自动生成了。点开之后（如下图）把这三个都打上对勾，点击 Agree 后，再执行 Process。

![confirm-agreement-mmd4mecanim](/images/confirm-agreement-mmd4mecanim.png)

随即会弹出来一个命令行窗口会进行 MMD 模型转化，执行一段时间后，可以看到 `Hoshifuri_Iku_MMD_v1.0_by_Itsuki_Fukurou` 文件夹下又多出来一堆东西。

![hoshifuri-folder-structure](/images/hoshifuri-folder-structure.png)

其中，`MMD Hoshifuri Iku v1.0` 是转化好的 MMD 模型预制件，直接拖进 Unity 场景里，震撼的一幕来了！

![mmd-does-not-works-on-unity6000.0.28f1c1](/images/mmd-does-not-works-on-unity6000.0.28f1c1.png)

何意味？！见到粉色的模型，第一时间就想到是不是 Material 或者 Shader 出了问题。

马上打开 Inspector 检查 `Hoshifuri_Iku_MMD_v1.0_by_Itsuki_Fukurou/Materials` 目录下的材质球，果不其然都变粉了！但是 Texture 确实都挂上了。

![pink-material-hsfrikuface](/images/pink-material-hsfrikuface.png)

仔细分析，由于本人使用的是 Unity 6000.0.28f1c1，是很高的版本。MMD 相关的各种插件都比较古老，在低版本 Unity 中一般使用的默认是 Built-in Render Pipeline，但是高版本早已经换成了 Universal Render Pipeline（URP）了。所以首先猜想，使用低版本的 Unity 2021.3.44f1c1 是不是可以正常渲染？

![mmd-works-well-on-unity2021.3.44](/images/mmd-works-well-on-unity2021.3.44.png)

嘿！还真是！那么第一个解决方案就很简单了，**即直接删除 Universal RP，在 Window > Packages Manage 中，找到 Universal RP 然后 Remove，最后根据提示重启 Unity Editor 即可。**

## 更优雅的解决方案？

但是还有个问题，那么假如我这个项目不允许我直接删掉 URP，怎么办？

首先，这里问题的关键就在于新版和旧版的 Shader 有很大差别，不兼容！所以要么是自己写一套新的 Shader，要么是通过某种方式将原来的 Shader 缝合到支持 URP 的 Shader 上，显然后者更简单。

好在，我很快就找到了这个由 ColinLeung-NiloCat 大佬编写的 Shader，链接如下

> 个人感觉这个 Shader 很适合二次元的那种风格！但如果你需要的是别的 Shader，解决方案可以继续往下看，是同理的。

[ColinLeung-NiloCat/UnityURPToonLitShaderExample: A very simple toon lit shader example, for you to learn writing custom lit shader in Unity URP](https://github.com/ColinLeung-NiloCat/UnityURPToonLitShaderExample)

用法很简单，只需要把仓库克隆下来到你的项目 Assets 目录下即可：

```
git clone https://github.com/ColinLeung-NiloCat/UnityURPToonLitShaderExample.git
```

接下来其实只需要把每个 MMD 中的 Material 中的 Shader 都换成这个新的 Shader 就好了。

具体地说，你需要点开某个 Material，在 Inspector 的上方可以看到 Shader `MMD4Mecanim/MMDLit-Edge`，点开小三角，然后搜索 **SimpleURPToonLitExample(With Outline)**，找到后点击即可切换。

![change-shader-mmd](/images/change-shader-mmd.png)

没完事呢，你发现切换 Shader 之后，Base Map 那里的 Texture 变成 None 了！此时**别忘记把原来的 Texture 放回来。**

![change-basemap-mmd](/images/change-basemap-mmd.png)

但我这一个二次元模型就有几十个 Material，一个一个改太折磨了，而且你很有可能漏选或者错选。所以我直接写了个脚本，将所有指定的 Material 的原 Texture 复制出来然后粘贴到新的 Material 上，需要的话直接抄：

```csharp
// -----------------------------------------------------------------------
//  <copyright file="MmdMaterialConverter.cs">
//  Copyright (c) 2025 AkagawaTsurunaki. All rights reserved.
//  Licensed under the MIT License.
//  </copyright>
//  <author>AkagawaTsurunaki (AkagawaTsurunaki@outlook.com)</author>
// -----------------------------------------------------------------------

using System.Collections.Generic;
using UnityEngine;

public class MmdMaterialConverter : MonoBehaviour
{
    [SerializeField] private List<Material> materials = new();

    private void Start()
    {
        foreach (var material in materials)
        {
            var mainTex = material.GetTexture("_MainTex");
            var toonTex = material.GetTexture("_ToonTex");
            material.shader = Shader.Find("SimpleURPToonLitExample(With Outline)");
            material.SetTexture("_BaseMap", mainTex);
            material.SetTexture("_ToonTex", toonTex);
        }
    }
}
```

用法是，将代码编写好后保存，挂在任意一个游戏对象上，然后在 Project 窗口中把你刚才 MMD 下的所有 Material 文件都勾选上（Ctrl + A），拖拽到 Inspector 的 `Materials` 上，注意一定是那个单词上，不要拖拽到下面的 List is empty 上，那不会起作用。

![drug-and-drop-materials](/images/drug-and-drop-materials.png)

> [!CAUTION]
>
> 这个脚本把指定的 Material 修改后是不能 Undo 撤销的，所以一定要备份！

随后运行整个项目，啊哈，出来了！

![elegant-mmd-unity6-solution-result](/images/elegant-mmd-unity6-solution-result.png)
