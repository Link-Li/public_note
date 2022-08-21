[TOC]


# PyTorch Lightning入门教程（一）

## 前言

因为最近在学习pytorch lightning，所以这里记录一下学习的内容，这一节记录简单的入门教程，下一节预计介绍如何进行多GPU训练。

pytorch lightning作为pytorch的一个拓展架构，可以减少很多与数据处理以及模型搭建无关的代码，增加工程效率。因为在编写训练代码的时候，很多时候需要兼顾编写学习率的schedule代码，记录log的代码等等。实际上，模型相关代码可能需要的时间还不如调试这些辅助的代码所需要的时间。而pytorch lightning这类框架就可以解决上面的问题。

虽然pytorch lightning框架有着很多优点，但是依旧有很多不足的地方，对于新手来说很不友好，尤其是官网的教程，虽然很丰富，但是感觉连续性比较差，如果对pytorch比较熟悉了，那可能看起来还行，否则非常的劝退。其次，。pytorch lightning每次版本更新迭代，可能都会修改它的一些接口，导致一些版本之间的兼容性可能会比较差。同时不同版本的pytorch lightning根据不同的pytorch版本开发，我常用的pytorch版本是1.7，导致pytorch lightning最高只能用到1.5.9。

本次的教程分为三部分，分别是安装，pytorch lightning简介和三个例子。

例子的代码见：https://github.com/Link-Li/pytorch-lightning-learn

## 安装

安装很方便，官方有介绍 https://pytorch-lightning.readthedocs.io/en/latest/starter/installation.html

但是这里非常不推荐用conda进行安装，因为conda可能安装不了自己需要的版本，我最初安装的版本是0.8的某个版本，导致很多接口和官网的教程都对不上。这里建议用pip进行安装，安装的时候注意要和pytorch的版本对应，不然安装的时候还会给你安装一个它需要的pytorch版本。

本教程使用的是pytorch lightning1.5.10版本，pytorch 1.7.1。其他版本的pytorch lightning可能接口会略有变化，请自己查看源代码说明。

## pytorch lightning结构简介


pytorch lightning的官网介绍了原生的pytorch和pytorch lightning的区别：

<!-- <video width="100%" max-width="800px" controls="" autoplay="" muted="" playsinline="" src="https://pl-bolts-doc-images.s3.us-east-2.amazonaws.com/pl_docs/pl_docs_animation_final.m4v"></video> -->

其实整体而言，就是将原来pytorch的模型部分（nn.Module）和训练逻辑（训练，验证，测试）部分结合在一起，组成了`pl.LightningModule`部分。之后采用`pl.Trainer`定义一个trainer接口，然后使用`trainer.fit`进行训练和验证，`trainer.test`进行预测。

### pl.LightningModule部分

如下所示，就是一个**简化**的pytorch lightning逻辑部分，我们需要定义一个类`CIFARModule`，然后继承自`pl.LightningModul`。

这里包含三部分，模型相关的部分`__init__`和`forword`；优化器相关的部分`configure_optimizers`；模型训练逻辑部分`training_step`,`validation_step`和`test_step`。
- 模型相关部分：这部分一般涉及到一些超参数的设定，模型的初始化以及具体的模型运行逻辑（forward函数）
- 优化相关部分：这部分一般涉及到模型的优化器初始化，学习率的schedule设定等
- 训练逻辑部分：这部分一般就是每个训练、验证、预测步骤需要做什么，除了这里列举的3个函数，pytorch lightning还提供了其他的很多的训练逻辑接口，在之后的例子中我们可以看到。

```
import pytorch_lightning as pl

class CIFARModule(pl.LightningModule):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, imgs):

    def configure_optimizers(self):

    def training_step(self, batch, batch_idx):

    def validation_step(self, batch, batch_idx):

    def test_step(self, batch, batch_idx):
```

### pl.Trainer部分

在使用`pl.LightningModule`定义好模型和训练逻辑之后，就需要定义trainer进行后续的训练和预测。

这里的`train_loader`可以使用pytorch原生的定义方式进行构造，对于`pl.Trainer`的参数，可以参考官方的API说明：https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html#pytorch_lightning.trainer.trainer.Trainer 

pytorch lightning提供了很多回调函数（callbacks），比如下面列举的LearningRateMonitor，可以记录学习率的变化，并绘制到tensorboard中，用于帮助确认学习率的schedule是否起作用了，此外还有很多其他的callbacks函数，可以参考官网的Api介绍：https://pytorch-lightning.readthedocs.io/en/latest/api_references.html#callbacks

在定义好trainer之后，就可以使用trainer的fit接口进行训练，test接口进行预测

```
train_loader, val_loader = get_data_loader()
model = CIFARModule()

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = pl.Trainer(callbacks=[lr_monitor], max_epochs=10, num_sanity_val_steps=2)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, dataloaders=val_loader, verbose=False)
```

## 举例说明

这里针对图像分类，文本分类和摘要生成三个任务进行举例分析。
详细代码见：https://github.com/Link-Li/pytorch-lightning-learn

### ResNet图像分类


这里使用CIFAR10数据集作为本次分类任务的数据集，采用ResNet-50作为骨干模型。

#### 数据准备

这里直接使用pytorch提供的CIFAR10数据集，并切分成训练集和验证集以及测试集

```
def get_data_loader(args):

    DATASET_PATH = "../data/"

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=8)

    return train_loader, val_loader, test_loader
```

#### 模型构建

这里直接使用pytorch提供的ResNet-50，然后采用继承`pl.LightningModule`的类`CIFARModule`来包裹真正的模型类`ResNet50`，这样的好处就是，不需要过多的修改我们之前习惯的模型代码的书写方式，只需要多定义一个类来适配到pytorch lightning框架。

针对`CIFARModule`，这里使用`self.save_hyperparameters()`来保存超参数，并在初始化函数中定义好损失函数和模型。之后在函数`configure_optimizers`中，定义好优化器和学习率的schedule，并返回定义好的优化器和schedule。这里的`configure_optimizers`返回值有多种形式，非常的灵活，具体的可以参考官网：https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers

之后在`training_step`，`validation_step`， `test_step`定义每个batch的训练逻辑，其中的`self.log`定义了tensorboard中记录日志的内容，具体的使用方式可以参考官网的教程：https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#log ，常用的应该就是name，value，on_step，on_epoch这些参数

```

class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.modle = resnet50(pretrained=True, progress=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        for param in self.modle.parameters():
            param.requires_grad = False
    
    def forward(self, imgs):
        return self.classifier(self.modle(imgs))


class CIFARModule(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.modle = ResNet50()
        self.loss = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        return self.modle(imgs)

    def configure_optimizers(self):
        optimizer = None
        scheduler = None
        if self.args.optimizer_name == "Adamw":
            optimizer = optim.AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.9)
        
        if self.args.scheduler_name == "lr_schedule":
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer, num_warmup_steps=self.args.warmup_step,
                num_training_steps=self.args.total_steps)

        if optimizer and scheduler:
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif optimizer:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train_acc", acc, on_step=True)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("val_acc", acc, on_step=True)
        self.log("val_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.modle(imgs)
        loss = self.loss(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("test_acc", acc, on_step=True)
        self.log("test_loss", loss, on_step=True)


```

#### 训练

这里也就是定义trainer接口的事情了，这里初始化`pl.Trainer`并没有使用直接传参的方式，而是采用`from_argparse_args`接口，将python的argparse模块的参数直接解析成`pl.Trainer`的参数。

同时这里定义了两个callbacks函数，其中一个`ModelCheckpoint`函数应该是用的比较多的一个callbacks函数，里面各种参数的说明可以参考：https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint

```

def train_model(args, model, train_loader, val_loader, test_loader):
    checkpoint = ModelCheckpoint(
        dirpath=args.checkpoint_path,
        filename="{epoch:02d}-{val_acc:.4f}",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint, lr_monitor])
    trainer.fit(model, train_loader, val_loader)
    print("trainer.checkpoint_callback.best_model_path: ", str(trainer.checkpoint_callback.best_model_path))

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result
```

### BERT文本分类

这里使用开源的数据集做一个简单的情感分类模型，整体代码和上面的ResNet-50类似，具体的可以参考https://github.com/Link-Li/pytorch-lightning-learn，这里就不做具体的分析了。

### T5摘要生成

#### 数据准备

这里使用了的一个开源的摘要生成的数据集：中文科学文献csl摘要数据，具体的代码分析这里就不再列举。除了下面预测结果的时候。

#### 预测

由于文本生成任务只看指标太抽象了，还是需要实际的看一下生成结果怎么样，所以我们需要将生成结果保存到一个文件中进行观察，这里需要修改`predict_step`和`on_predict_batch_end`两个函数来保存生成结果。

```
class CIFARModule(pl.LightningModule):
    def __init__(self, args, tokenizer) -> None:
        super().__init__()
        ... ...

    def forward(self, imgs):
        ... ...

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        input_ids, attention_mask, labels = batch
        labels[labels == -100] = self.tokenizer.pad_token_id
        preds = self.model.generate(input_ids, attention_mask)
        preds_text = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return {"pre": preds_text, "source": input_text, "target": labels_text}

    def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        with open(self.args.save_file_path, "a", encoding="utf-8") as f_write:
            temp_save = {}
            for pre, source, target in zip(outputs["pre"], outputs["source"], outputs["target"]):
                temp_save["source"] = source
                temp_save["target"] = target
                temp_save["pre"] = pre
                f_write.write(json.dumps(temp_save, ensure_ascii=False))
```

这里列举几条生成结果，似乎还可以：

```
{
    "source": "采用通用可组合的方法,首次提出了数字签名的框架体系,根据数字签名的特点,在数字签名框架体系下划分成六大模块,将这些模块有机组合,对数字签名进行分类。这种分类方法有利于对已有的数字签名的研究,也有助于对新签名的探索研究。",
    "target": "通用可组合数字签名框架体系",
    "pre": "通用可组合的数字签名框架体系"
}
{
    "source": "为了利用依存关系进行短文本分类,研究了利用依存关系进行短文本分类存在的四个关键问题。分别在长文本语料集和两个短文本语料集上,抽取具有依存关系的词对,并利用这些词对作为特征进行分类实验。实验结果表明:依存关系能够作为有效的特征进行文本分类,并能够改善文本分类的性能;单独把依存关系作为特征,不能提高短文本的分类性能;可以利用依存关系作为特征扩充的手段,增加短文本的特征,增强短文本的描述能力,进而进行有效的短文本分类。",
    "target": "中文文本分类中利用依存关系的实验研究",
    "pre": "依存关系下基于依存关系的短文本分类研究"
}
```

下一个教程应该会讲解多GPU训练的相关代码