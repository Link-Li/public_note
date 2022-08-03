- 查看目前正在运行的docker

```
docker ps
```

- 查看所有的docker

```
docker ps -a
```

- 查看所有的docker的id
```
docker ps -aq
```

- 停止所有的docker

```
docker stop $(docker ps -aq)
```

- 删除所有的docker

```
docker rm $(docker ps -aq)
```

- 列出所有的镜像

```
docker images
```

- 删除对应的docker

```
docker rmi images_id
```

- 临时进入docker

```
docker run -it docker_name_ubuntu /bin/bash
```
-i表示交互式运行docker
-t表示为docker分配一个终端

- 用docker运行代码

```
docker run -d -v /data:/data docker_name /bin/bash /data/code/test_run.sh
```
其中v是将本地文件夹挂载到docker里面，其中本地目录在前，docker目录在后