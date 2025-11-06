# 如何在Windows上通过VNC远程桌面连接到Ubuntu主机？

> Ciallo～(∠・ω< )⌒★ 我是赤川鹤鸣。最近需要用到图形界面，但是来回把Ubuntu主机插到显示器上太麻烦了。了解到有VNC这种方便快捷的东西，踩了一些坑，终于安装成功了。

### Ubuntu 上安装 TigerVNC

这里假设你已经在一台主机上安装好了 Ubuntu 系统，我的 Ubuntu 版本是 22.04.5。

首先，需要安装一个桌面环境。个人比较喜欢 gnome 的风格，这里就以 gnome 为例了。

``` shell
sudo apt update
sudo apt install ubuntu-gnome-desktop
```

接着，安装TigerVNC以及必要的依赖。

``` shell
sudo apt install tigervnc-standalone-server tigervnc-xorg-extension tigervnc-viewer
```

然后，咱们需要配置一下 VNC 的配置文件。这里我习惯用 nano 进行配置。 

``` shell
nano ~/.vnc/xstartup
```

需要注意的是，这个配置文件是属于**“当前登录用户”**的，只对启动 VNC 时的那个 Linux 帐号生效。

在 ~/.vnc/xstartup 中输入下面这大段代码。这段是我问 Kimi 得到的，出奇地好用。

``` bash
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS
export XDG_SESSION_TYPE=x11
export XDG_CURRENT_DESKTOP=ubuntu:GNOME   # 或 GNOME，根据发行版
export GNOME_SHELL_SESSION_MODE=ubuntu    # Ubuntu 需加这一行
exec gnome-session --session=ubuntu       # 纯 Debian 用 gnome-session
```

之后保存文件 **Ctrl+O**，退出文本编辑器 **Ctrl+X**。

配置文件写完后，别忘了**修改权限**，防止运行时报错

``` shell
chmod +x ~/.vnc/xstartup
```

接着启动 vncserver，

``` shell
vncserver -localhost no
```

其中，-localhost no 表示暴露给外部可访问，如果改写成 yes 那么只有本机地址可以访问。因为我有公网IP，所以就设为 no。 

如果你是第一次使用，VNC 可能需要你为访问者设置密码，为了保证安全还是建议设置一个强密码（当然得自己记得住）。

上述操作完毕后，可以在控制台的回显上看到开启的端口。

``` shell
New Xtigervnc server 'xmm-ubuntu:1 (xmm)' on port 5901 for display :1.
Use xtigervncviewer -SecurityTypes VncAuth,TLSVnc -passwd /home/xmm/.vnc/passwd xmm-ubuntu:1 to connect to the VNC server.
```

可以看到，**显示器 1 号开启于端口 5901。请记住这个端口号！**

### Windows 上安装 TigerVNC

在 Windows 上下载并安装一个 TigerVNC Viewer，第一个链接是源代码仓库，**第二个链接是直装版的安装包**。

安装完毕后，打开 TigerVNC Viewer，在弹出的对话框中输入你的 Ubuntu 主机 IP 和端口。

<img src="/images/tigervncviewer-connection-details.png" width="500" alt="TigerVNC Viewer">

添加图片注释，不超过 140 字（可选）

这里 xxxxxx 是**主机的 IP（记得换成你自己的）**，**端口号**就是刚才执行命令后给出的。

确认好地址后，点击**连接**，接着**输入密码**后就可以了。

### 其他问题

如果想要关闭刚才启动的 VNC 服务器，我们必须 kill 对应的显示器。刚才我们是显示器 1 号被启动了，所以

``` shell
vncserver -kill :1
```

就可以了。

进入登录界面，输入密码后卡死，请看

- [用vncviewer登服务器界面卡住操作步骤](https://zhuanlan.zhihu.com/p/671888731)
- [AskUbuntu - I cannot log in a vnc session after the screen locks authentification error](https://askubuntu.com/questions/1224957/i-cannot-log-in-a-vnc-session-after-the-screen-locks-authentification-error)