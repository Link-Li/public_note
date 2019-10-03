

# ssh快速连接

&emsp;&emsp;在window上面, 我们可以使用的ssh管理软件比较多. 但是在Ubuntu上面, 可以使用的比较少, 有pac manager, 还有国产软件finalShell. 但是在实际使用的时候, 我安装的pac无法启动, 或许是因为依赖问题没有搞定. finalShell莫名占用一个核的cpu. 所以最后还是回归到了Ubuntu上面的openssh.

&emsp;&emsp;在直接使用Ubuntu上面的ssh的时候, 每次我们都要输入下面的命令

```
ssh username@ip_address
```

&emsp;&emsp;偶尔使用还行, 但是当有多个主机并且需要多次连接的时候, 就很麻烦了. 偶尔看到有人说可以在ssh中配置好快捷方式, 然后直接输入`ssh name`就行了. 经过查找资料, 最后得到了完整的使用方式. 

&emsp;&emsp;为了说明方便, 客户机用C表示, 远程主机用S表示.

## 首先创建config文件

&emsp;&emsp;在C上面, 进入到`~/.ssh`文件夹下, 如果没有`config`文件, 就创建一个:

```
cd ~/.ssh
sudo vim config
```

&emsp;&emsp;然后在`config`文件中输入:

```
Host you_name
HostName 10.10.10.1
User username
IdentityFile xxx/xxx/xxx/id_rsa
```

&emsp;&emsp;注意上面每一行的后缀改成你自己的信息, 而且在`config`文件中你可以随意增加上面这样的语句, 每个都对应一个远程主机.

## 添加私钥和公钥

&emsp;&emsp;其实上面的那四句中的最后一句, 不添加也是可以的, **如果不添加的话**, 按照上面的文件中的设置, 你只需要输入

```
ssh you_name
```

&emsp;&emsp;然后就可以连接到远程的主机了, 但是这个时候需要输入密码. 

&emsp;&emsp;为了不在输入密码, 我们就需要创建私钥和公钥, 私钥是放在C机器上的, 公钥是放在S机器上的.

&emsp;&emsp;创建ssh的私钥和公钥的话, 在C机器上输入下面的命令:

```
ssh-keygen -t rsa  
```

&emsp;&emsp;输入之后, 首先会让你选择生成的公钥和私钥的位置, 你可以选择默认, 也可以自己找一个地方存, 但是前提是你得先建好文件夹, 然后才能在这里选择存在你建好的文件夹的位置, 否则会报错说找不到对应的文件夹. 然后会让你输入一些验证密码啥的, 这个可以不输入, 直接回车跳过就行了.

&emsp;&emsp;生成好密钥之后, 我们在C机器上输入下面的代码, 其中ip是S机器的ip:

```
ssh-copy-id -i 私钥的地址 username@ip_address

例如:
ssh-copy-id -i ~/.ssh/username/id_rsa username@10.10.10.1
```

&emsp;&emsp;然后会让你输入S机器上面的密码, 输入密码之后, 这个时候就算完成所有的工作了, 你这个时候再输入命令`ssh you_name`就可以直接连接上远程的主机了.





