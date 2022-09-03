[TOC]


## 相关命令总结

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

- 开启一个docker环境后台运行

```
nvidia-docker run -itd -e NVIDIA_VISIBLE_DEVICES=all --network=host -v /data/:/data/ docker_name /bin/bash
```

- 进入开启的docker

```
nvidia-docker exec -it container_id /bin/bash
```


## 如何在Docker中安装anaconda

使用如下的命令：

```
FROM ubuntu-python3.8.9:latest
COPY Miniconda3-py38_4.12.0-Linux-x86_64.sh /opt
RUN cd /opt \
&& sh Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p /opt/conda \
&& rm -rv Miniconda3-py38_4.12.0-Linux-x86_64.sh \
&& sed -i "10aPATH=$PATH:/opt/conda/bin" ~/.bash_profile \
&& source ~/.bash_profile \
&& conda init \
&& rm -rv /usr/bin/python3 \
&& ln -s /opt/conda/bin/python /usr/bin/python3 \
&& conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch \
&& pip install transformers==4.18.0 pytorch-lightning==1.5.10 fairscale==0.4.2 nltk==3.6.7 rouge==1.0.1 sentencepiece==0.1.96
```