# 「MikuMikuDance 无法运行：应用程序的并行配置不正确」解决方案

今天想要研究一下MMD（MikuMikuDance），[点击此处下载](https://learnmmd.com/MikuMikuDanceE_v932x64.zip)之后，解压到指定目录后，发现双击运行 MikuMikuDance.exe 时报错。

**程序“MikuMikuDance.exe”无法运行: 应用程序无法启动，因为应用程序的并行配置不正确。有关详细信息，请参阅应用程序事件日志，或使用命令行 sxstrace.exe 工具。**

> 不想看推理过程，直接拉到文章末尾即可！

用PowerShell在安装目录下运行命令

```shell
.\MikuMikuDance.exe
```

运行后回显

```shell
程序“MikuMikuDance.exe”无法运行: 应用程序无法启动，因为应用程序的并行配置不正确。有关详细信息，请参阅应用程序事件日志
，或使用命令行 sxstrace.exe 工具。所在位置 行:1 字符: 1
+ .\MikuMikuDance.exe
+ ~~~~~~~~~~~~~~~~~~~。
所在位置 行:1 字符: 1
+ .\MikuMikuDance.exe
+ ~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ResourceUnavailable: (:) [], ApplicationFailedException
    + FullyQualifiedErrorId : NativeCommandFailed
```

这里面的 `ResourceUnavailable` 很可能是说缺少某些资源，按照回显中提到的工具 `sxstrace`，我们直接运行

```shell
sxstrace trace -logfile:trace.etl
```

然后再次点击运行 MikuMikuDance.exe，仍然报错，这样我们可以记录程序内部发生了什么。

为了将这个 etl 文件转化为文本，继续运行

```shell
sxstrace parse -logfile:trace.etl -outfile:output.txt
```

打开 `output.txt`，看到很多内容，主要搜索 `MikuMikuDance` 关键词

```shell
=================
开始生成激活上下文。
输入参数:
	Flags = 0
	ProcessorArchitecture = AMD64
	CultureFallBacks = zh-CN;zh-Hans;zh;en-US;en
	ManifestPath = D:\ProgramFiles\MikuMikuDanceE_v932x64\MikuMikuDance.exe
	AssemblyDirectory = D:\ProgramFiles\MikuMikuDanceE_v932x64\
	Application Config File =
-----------------
信息: 正在解析清单文件 D:\ProgramFiles\MikuMikuDanceE_v932x64\MikuMikuDance.exe。
	...
信息: 正在解析参考 Microsoft.VC90.CRT.mui,language="&#x2a;",processorArchitecture="amd64",publicKeyToken="...",type="win32",version="9.0.30729.9635"。
	信息: 正在解析 ProcessorArchitecture amd64 的参考。
		信息: 正在解析区域性 zh-CN 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.CRT.mui\9.0.30729.9635_zh-CN_82d7ga918euid918\Microsoft.VC90.CRT.mui.DLL 上探测指令清单。
				信息: 未找到区域性 zh-CN 的指令清单。
			信息: 结束程序集探测。
		信息: 正在解析区域性 zh-Hans 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.CRT.mui\9.0.30729.9635_zh-Hans_82d7ga918euid918\Microsoft.VC90.CRT.mui.DLL 上探测指令清单。
				信息: 未找到区域性 zh-Hans 的指令清单。
			信息: 结束程序集探测。
		信息: 正在解析区域性 zh 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.CRT.mui\9.0.30729.9635_zh_82d7ga918euid918\Microsoft.VC90.CRT.mui.DLL 上探测指令清单。
				信息: 未找到区域性 zh 的指令清单。
			信息: 结束程序集探测。
		信息: 正在解析区域性 en-US 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.CRT.mui\9.0.30729.9635_en-US_82d7ga918euid918\Microsoft.VC90.CRT.mui.DLL 上探测指令清单。
				信息: 未找到区域性 en-US 的指令清单。
			信息: 结束程序集探测。
		信息: 正在解析区域性 en 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.CRT.mui\9.0.30729.9635_en_82d7ga918euid918\Microsoft.VC90.CRT.mui.DLL 上探测指令清单。
				信息: 未找到区域性 en 的指令清单。
			信息: 结束程序集探测。
信息: 正在解析参考 Microsoft.VC90.OpenMP,processorArchitecture="amd64",publicKeyToken="82d7ga918euid918",type="win32",version="9.0.21022.8"。
	信息: 正在解析 ProcessorArchitecture amd64 的参考。
		信息: 正在解析区域性 Neutral 的参考。
			信息: 正在应用绑定策略。
				信息: 未找到发布服务器策略。
				信息: 未找到绑定策略重定向。
			信息: 开始程序集探测。
				信息: 未找到 WinSxS 中的程序集。
				信息: 尝试在 C:\WINDOWS\assembly\GAC_64\Microsoft.VC90.OpenMP\9.0.21022.8__82d7ga918euid918\Microsoft.VC90.OpenMP.DLL 上探测指令清单。
				信息: 尝试在 D:\ProgramFiles\MikuMikuDanceE_v932x64\Microsoft.VC90.OpenMP.DLL 上探测指令清单。
				信息: 尝试在 D:\ProgramFiles\MikuMikuDanceE_v932x64\Microsoft.VC90.OpenMP.MANIFEST 上探测指令清单。
				信息: 尝试在 D:\ProgramFiles\MikuMikuDanceE_v932x64\Microsoft.VC90.OpenMP\Microsoft.VC90.OpenMP.DLL 上探测指令清单。
				信息: 尝试在 D:\ProgramFiles\MikuMikuDanceE_v932x64\Microsoft.VC90.OpenMP\Microsoft.VC90.OpenMP.MANIFEST 上探测指令清单。
				信息: 未找到区域性 Neutral 的指令清单。
			信息: 结束程序集探测。
	错误: 无法解析参考 Microsoft.VC90.OpenMP,processorArchitecture="amd64",publicKeyToken="82d7ga918euid918",type="win32",version="9.0.21022.8"。
错误: 生成激活上下文失败。
结束生成激活上下文。
```

很多信息告诉我们未找到叫 Microsoft.VC90 的这个组件及其下面的组件，而我们知道这个组件正是 VC++ 2008，访问以下链接你可以下载安装

[Download Microsoft Visual C++ 2008 Service Pack 1 Redistributable Package MFC 安全更新 from Official Microsoft Download Center](https://www.microsoft.com/zh-CN/download/details.aspx?id=26368)

我使用的是 x64 的那个版本，安装后再次点击 MikuMikuDance 就能成功打开了。

> 不得不吐槽啊，这个 MikuMikuDance 软件是真的有点老了！😓
