@[TOC](目录)

pytorch中如何在lstm中输入可变长的序列
=================

   * [pytorch中如何在lstm中输入可变长的序列](#pytorch中如何在lstm中输入可变长的序列)
      * [torch.nn.utils.rnn.pad_sequence()](#torchnnutilsrnnpad_sequence)
      * [torch.nn.utils.rnn.pack_padded_sequence()](#torchnnutilsrnnpack_padded_sequence)
      * [torch.nn.utils.rnn.pad_packed_sequence()](#torchnnutilsrnnpad_packed_sequence)


<br>
<br>
# pytorch中如何在lstm中输入可变长的序列

我在做的时候主要参考了这些文章
<a href="https://zhuanlan.zhihu.com/p/59772104" target="_blank">https://zhuanlan.zhihu.com/p/59772104</a>
<a href="https://blog.csdn.net/u011550545/article/details/89529977" target="_blank">https://blog.csdn.net/u011550545/article/details/89529977</a>

&emsp;&emsp;前两天在做一个情感二分类任务的时候, 使用了lstm. 但是在将句子输入lstm的时候, 发现句子的分布相当的分散, 长度从600个单词到3个单词不等. 本来打算直接截断输入到lstm中, 但是对训练数据分析了一下, 发现数据的分布不是特别的集中, 实际使用截断的方法做的话, 和使用不截断的方法, 差了大概3%左右. 那么pytorch如何使用变长的序列输入呢?

&emsp;&emsp;这里给出其中最主要的3个方法

```
torch.nn.utils.rnn.pad_sequence()
torch.nn.utils.rnn.pack_padded_sequence()
torch.nn.utils.rnn.pad_packed_sequence()
```

其中`torch.nn.utils.rnn.pad_sequence()`把不等长的tensor数据, 补充成等长的tensor数据.

`torch.nn.utils.rnn.pack_padded_sequence()`把等长的tensor根据所输入的参数压缩成实际的数据, 同时数据格式变成`PackedSequence`

`torch.nn.utils.rnn.pad_packed_sequence()`把上面所压缩成`PackedSequence`的数据还原成tensor类型, 并补成等长的数据, 下面依次介绍一下.

## torch.nn.utils.rnn.pad_sequence()

&emsp;&emsp;首先是`torch.nn.utils.rnn.pad_sequence()`, 我们看一个tensor数据

```
train_x = [torch.tensor([1, 2, 3, 4, 5, 6, 7]),
           torch.tensor([2, 3, 4, 5, 6, 7]),
           torch.tensor([3, 4, 5, 6, 7]),
           torch.tensor([4, 5, 6, 7]),
           torch.tensor([5, 6, 7]),
           torch.tensor([6, 7]),
           torch.tensor([7])]
```

可以明显的看到,这个数据是不等长的, 这样的话, 我们是无法将这些数据组成一组数据送入网络进行训练的, 例如我们写一个简单的数据读取的:

```
class MyData(Dataset):
    def __init__(self, train_x):
        self.train_x = train_x

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, item):
        return self.train_x[item]
        
train_data = MyData(train_x)
train_dataloader = DataLoader(train_data, batch_size=2)

for i in train_dataloader:
    print(i)
```

结果我们得到的是:

```
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 7 and 6 in dimension 1 at /tmp/pip-req-build-58y_cjjl/aten/src/TH/generic/THTensor.cpp:689
```

&emsp;&emsp;这个说明, 如果两个数据不一样长, 是没法进行拼接的, 这样的话我们就没法进行批量的数据处理了, 我们需要做的就是将每个批次的数据填充成一样长的, 但是我们要怎么做呢? 在`__getitem__(self, item)`方法中填充吗?

&emsp;&emsp;在`__getitem__(self, item)`中填充确实是可以的, 但是这样做的方法有点不好, 就是我们每次进行批次训练的时候, 我们每个批次的数据中最长的那个数据不一定每次都是一样的, 如果在`__getitem__(self, item)`中进行填充, 我们就要取所有数据集中最长的那个作为基准, 然后把所有的数据都填充成那么长的. 但是这样做肯定是不合适的. 实际上, pytorch中的`DataLoader`提供了一个参数`collate_fn=`, 通过这个参数, 我们可以传入一个函数或者类, 进行数据的处理.

&emsp;&emsp;`collate_fn`参数就是进行处理在选出所需要的数据之后, 如何把这些数据拼接成一个整体的tensor. 平时在默认使用的时候, 这个参数的默认函数会直接把所取到的数据拼接成一个完整的tensor, 但是现在我们的数据是不等长的, 那么在拼接的时候肯定会出问题, 那么我们需要做的其实就是自定义一个`collate_fn`函数, 然后我们来拼接数据, 例如定义一个下面的`collate_fn`函数

```
def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data, data_length
```

这里我返回了两个参数, 一个是`train_data`, 另外一个是`data_length`. 而且我还对`train_data`进行了一次排序. 这些一会解释. 我们现在需要知道的是, 我们已经对`train_data`进行了填充, 并将它合并成了一个完整的tensor返回, 这个时候, 我们再把数据传入`DataLoader`中

```
train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)

for data, length in train_dataloader:
    print(data)
    print(length)
```

然后我们会看到这样的输出

```
tensor([[1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 0]])
[7, 6]
tensor([[3, 4, 5, 6, 7],
        [4, 5, 6, 7, 0]])
[5, 4]
tensor([[5, 6, 7],
        [6, 7, 0]])
[3, 2]
tensor([[7]])
[1]
```

## torch.nn.utils.rnn.pack_padded_sequence()

&emsp;&emsp;这个时候, `pad_sequence`的作用也就讲完了, 下面就是`pack_padded_sequence`. `pack_padded_sequence`函数的字面意思就是把原来填充过的序列再压缩回去. 它有三个主要的参数, 分别是`input, lengths, batch_first`. 其中`input`就是我们上面使用`pad_sequence`填充过的数据, 而`lengths`就是我们`collate_fn`函数返回的`length`, 也就是我们的数据的实际长度, `batch_first`就简单了, 就是把数据的`batch_first`放到最前面.

&emsp;&emsp;但是为啥我们需要使用`pack_padded_sequence`呢? 直接把填充好的数据输入到`RNN`中不可以吗?实际上是当然可以的, 但是在实际情况中, 数据是这样输入的, 下面给出一个batch的例子

```
tensor([[1, 2, 3, 4, 5, 6, 7],
        [2, 3, 4, 5, 6, 7, 0]])
```

输入到`RNN`的实际上是按照这样的顺序`[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]`依次输入到`RNN`中的. 但是我们发现最后一个是`[7, 0]`, 这里的0输入到`RNN`中, 实际上并没有输出有用的数据, 这样的话就会浪费算力资源, 所以我们使用pack_padded_sequence进行压缩一下, 例如下面的代码:

```
for data, length in train_dataloader:
    data = rnn_utils.pack_padded_sequence(data, length, batch_first=True)
    print(data)
```

输出的结果是:

```
PackedSequence(data=tensor([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]), batch_sizes=tensor([2, 2, 2, 2, 2, 2, 1]), sorted_indices=None, unsorted_indices=None)
PackedSequence(data=tensor([3, 4, 4, 5, 5, 6, 6, 7, 7]), batch_sizes=tensor([2, 2, 2, 2, 1]), sorted_indices=None, unsorted_indices=None)
PackedSequence(data=tensor([5, 6, 6, 7, 7]), batch_sizes=tensor([2, 2, 1]), sorted_indices=None, unsorted_indices=None)
PackedSequence(data=tensor([7]), batch_sizes=tensor([1]), sorted_indices=None, unsorted_indices=None)
```

&emsp;&emsp;我们可以只看第一个输出, 这个是我们上面举的那个例子, 数据类型已经变成了`PackedSequence`, 同时数据变成了`tensor([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])`, 你看着肯定很奇怪, 这样的数据还怎么输入到`RNN`中进行运算呢? 

&emsp;&emsp;但是`PackedSequence`还提供了一个`batch_sizes`数据, 这个数据其实是用来分割前面的那一串数据的, 例如`batch_sizes`前面6个都是2, 最后一个是1, 和上面的`[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]`是对应着的, 只是最后一个0没有了, 所以`batch_sizes`最后一个变成了1, 其实就相当于把数据分割成了`[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7]`. 
&emsp;&emsp;`pack_padded_sequence`所起的作用, 其实有点类似于重新对数据的batch_size进行了修改, 根据数据的实际长度, 将每次输入的数据的batch_size值进行修改, 这样也就完成了可变长数据的输入

## torch.nn.utils.rnn.pad_packed_sequence()
&emsp;&emsp;`pad_packed_sequence`又是干什么用的呢? 我们可以看一下, 将上面的可变长序列输入`LSTM`之后的输出是什么. 我们先定义一个`LSTM`

```
net = nn.LSTM(1, 5, batch_first=True)
```

&emsp;&emsp;定义的这个`LSTM`是输入的维度是1维, 输出的维度是5维的. 但是它的完整数据输入格式是`input (batch, seq_len, input_size)`. 和我们上面举例的数据稍微有点不太一样, 所以我们需要修改一下`collate_fn`函数, 让它符合我们的输入要求:

```
def collate_fn(train_data):
    train_data.sort(key=lambda data: len(data), reverse=True)
    data_length = [len(data) for data in train_data]
    train_data = rnn_utils.pad_sequence(train_data, batch_first=True, padding_value=0)
    return train_data.unsqueeze(-1), data_length  # 对train_data增加了一维数据
```

然后我们将数据输入`LSTM`, 看一下输出的第一次的结果:

```
train_data = MyData(train_x)
train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)

flag = 0
for data, length in train_dataloader:
    data = rnn_utils.pack_padded_sequence(data, length, batch_first=True)
    output, hidden = net(data)
    if flag == 0:
        print(output)
        flag = 1
```

```
out:
PackedSequence(data=tensor([[-0.0359, -0.0036,  0.0825,  0.1019, -0.1004],
        [ 0.0155,  0.0222,  0.0926,  0.1369, -0.0548],
        [-0.0054,  0.0196,  0.1241,  0.1759, -0.1449],
        [ 0.0495,  0.0504,  0.1263,  0.2017, -0.0534],
        [ 0.0374,  0.0475,  0.1405,  0.2131, -0.1426],
        [ 0.0729,  0.0720,  0.1338,  0.2225,  0.0114],
        [ 0.0656,  0.0693,  0.1410,  0.2237, -0.0812],
        [ 0.0792,  0.0866,  0.1280,  0.2228,  0.1560],
        [ 0.0743,  0.0844,  0.1319,  0.2203,  0.0601],
        [ 0.0737,  0.0962,  0.1156,  0.2154,  0.3757],
        [ 0.0701,  0.0946,  0.1179,  0.2117,  0.2878],
        [ 0.0630,  0.1021,  0.1004,  0.2058,  0.6103],
        [ 0.0604,  0.1011,  0.1020,  0.2022,  0.5502]], grad_fn=<CatBackward>), batch_sizes=tensor([2, 2, 2, 2, 2, 2, 1]), sorted_indices=None, unsorted_indices=None)
```

这个结果是`PackedSequence`类型的, 而且数据格式是[13 * 5]的, 如果我们下面需要经过一个全连接层, 那么我们需要的数据格式应该是[2 * 7 * 5]的形式. 这个时候, 就是`pad_packed_sequence`发挥作用的时候了. 我们看下面的代码:

```
train_data = MyData(train_x)
train_dataloader = DataLoader(train_data, batch_size=2, collate_fn=collate_fn)

flag = 0
for data, length in train_dataloader:
    data = rnn_utils.pack_padded_sequence(data, length, batch_first=True)
    output, hidden = net(data)
    if flag == 0:
        output, out_len = rnn_utils.pad_packed_sequence(output, batch_first=True)
        print(output.shape)
        print(output)
        flag = 1
```

可以看到输出结果是:

```
torch.Size([2, 7, 5])
tensor([[[-0.0359, -0.0036,  0.0825,  0.1019, -0.1004],
         [-0.0054,  0.0196,  0.1241,  0.1759, -0.1449],
         [ 0.0374,  0.0475,  0.1405,  0.2131, -0.1426],
         [ 0.0656,  0.0693,  0.1410,  0.2237, -0.0812],
         [ 0.0743,  0.0844,  0.1319,  0.2203,  0.0601],
         [ 0.0701,  0.0946,  0.1179,  0.2117,  0.2878],
         [ 0.0604,  0.1011,  0.1020,  0.2022,  0.5502]],

        [[ 0.0155,  0.0222,  0.0926,  0.1369, -0.0548],
         [ 0.0495,  0.0504,  0.1263,  0.2017, -0.0534],
         [ 0.0729,  0.0720,  0.1338,  0.2225,  0.0114],
         [ 0.0792,  0.0866,  0.1280,  0.2228,  0.1560],
         [ 0.0737,  0.0962,  0.1156,  0.2154,  0.3757],
         [ 0.0630,  0.1021,  0.1004,  0.2058,  0.6103],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000]]],
       grad_fn=<TransposeBackward0>)
```

这里我们可以看到, 输出的数据符合[2 * 7 * 5]的形式, 并且看实际数据, 可看到最后一个数据的不足补的都是0.

<a href="https://github.com/Link-Li/blog_others/blob/master/pytorch_lstm_change_sequence.ipynb"  target="_blank">博客中代码参考pytorch_lstm_change_sequence.ipynb</a>