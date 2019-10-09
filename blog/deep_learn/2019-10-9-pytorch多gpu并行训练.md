
# pytorch多gpu并行训练

*暂时只是使用了单机多卡的GPU进行测试, 并没有使用多机多卡, 这里只简述了如何使用DistributedDataParallel代替DataParallel*

## torch.nn.DataParallel

&emsp;&emsp;我一般在使用多GPU的时候, 会喜欢使用`os.environ['CUDA_VISIBLE_DEVICES']`来限制使用的GPU个数, 例如我要使用第0和第3编号的GPU, 那么只需要在程序中设置:

```
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3'
```

&emsp;&emsp;但是要注意的是, 这个参数的设定要保证在模型加载到gpu上之前, 我一般都是在程序开始的时候就设定好这个参数, 之后如何将模型加载到多GPU上面呢?

&emsp;&emsp;如果是模型, 那么需要执行下面的这几句代码:

```
model = nn.DataParallel(model)
model = model.cuda()
```

&emsp;&emsp;如果是数据, 那么直接执行下面这几句代码就可以了:

```
inputs = inputs.cuda()
labels = labels.cuda()
```

&emsp;&emsp;其实如果看pytorch<a href
='https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html' traget="_blank">官网</a>给的示例代码,我们可以看到下面这样的代码

```
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

model.to(device)
```

&emsp;&emsp;这个和我上面写的好像有点不太一样, 但是如果看一下`DataParallel`的内部代码, 我们就可以发现, 其实是一样的:

```
class DataParallel(Module):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
```

&emsp;&emsp;我截取了其中一部分代码, 我们可以看到如果我们不设定好要使用的`device_ids的`话, 程序会自动找到这个机器上面可以用的所有的显卡, 然后用于训练. 但是因为我们前面使用`os.environ['CUDA_VISIBLE_DEVICES']`限定了这个程序可以使用的显卡, 所以这个地方程序如果自己获取的话, 获取到的其实就是我们上面设定的那几个显卡.

&emsp;&emsp;我没有进行深入得到考究, 但是我感觉使用`os.environ['CUDA_VISIBLE_DEVICES']`对可以使用的显卡进行限定之后, 显卡的实际编号和程序看到的编号应该是不一样的, 例如上面我们设定的是`os.environ['CUDA_VISIBLE_DEVICES']="0,2"`, 但是程序看到的显卡编号应该被改成了```'0,1'```, 也就是说程序所使用的显卡编号实际上是经过了一次映射之后才会映射到真正的显卡编号上面的, 例如这里的程序看到的1对应实际的2

## torch.nn.parallel.DistributedDataParallel

&emsp;&emsp;pytorch的官网建议使用`DistributedDataParallel`来代替`DataParallel`, 据说是因为`DistributedDataParallel`比`DataParallel`运行的更快, 然后显存分屏的更加均衡. 而且`DistributedDataParallel`功能更加强悍, 例如分布式的模型(一个模型太大, 以至于无法放到一个GPU上运行, 需要分开到多个GPU上面执行). 只有`DistributedDataParallel`支持分布式的模型像单机模型那样可以进行多机多卡的运算.当然具体的怎么个情况, 建议看官方文档. 对于我而言, 在可预见的未来不会使用多机多卡来训练, 单机多卡已经可以满足我的需求了, 这里只记录一下如何使用`DistributedDataParallel`代替`DataParallel`

&emsp;&emsp;依旧是先设定好`os.environ['CUDA_VISIBLE_DEVICES']`, 然后再进行下面的步骤.

&emsp;&emsp;因为`DistributedDataParallel`是支持多机多卡的, 所以这个需要先初始化一下, 如下面的代码:

```
torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
```

&emsp;&emsp;第一个参数是pytorch支持的通讯后端, 不懂, 但是据说nccl的支持最好, 但是这里单机多卡, 这个就是走走过场. 第二个参数是各个机器之间通讯的方式, 可以选择共文件的方式, 或者选择这里的TCP协议的方式, 后面的ip自己选一个吧, 我用localhost没报错, 用的别的地址有的报错, 但是依旧没影响, 后面的端口自己找一个空着的就行了. rank是标识主机和从机的, 这里就一个主机, 设置成0就行了. world_size是标识使用几个GPU的, 但是我只有设置成1的时候可以正常运行, 很迷, 不知道为啥, 但是不影响我使用单机多卡, 更迷了.

&emsp;&emsp;之后就和使用`DataParallel`很类似了.

```
model = emotion_net.cuda()
model = nn.parallel.DistributedDataParallel(model)
```

&emsp;&emsp;但是注意这里要先将`model`加载到GPU, 然后才能使用`DistributedDataParallel`进行分发, 之后的使用和`DataParallel`就基本一样了