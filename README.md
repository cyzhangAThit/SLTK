# SLF - Sequence Labeling Framework

序列化标注框架，实现了Bi-LSTM-CRF模型，并利用pytorch实现了高效的数据加载模块，可以完成:

 - **预处理**。包括构建词表、label表，从预训练文件构建word embedding;
 - **训练**。训练序列化标注模型，并保存在开发集上性能最好的一次模型;
 - **测试**。对新的实例进行标注。

## 1. 快速开始

### 1.1 数据格式

训练数据处理成下列形式，特征之间用制表符(或空格)隔开，每行共n列，1至n-1列为特征，最后一列为label，句子之间用**空行**分隔。

    苏   NR   B-ORG
    州   NR   I-ORG
    大   NN   I-ORG
    学   NN   E-ORG
    位   VV   O
    于   VV   O
    江   NR   B-GPE
    苏   NR   I-GPE
    省   NR   E-GPE
    苏   NR   B-GPE
    州   NR   I-GPE
    市   NR   E-GPE

### 1.2 预处理&训练&测试

Step 1:

将训练、测试文件处理成所需格式，放入`../data/`目录下，文件名分别为`train.txt`和`test.txt`。

Step 2:

    $ chmod a+x *.sh

Step 3:

    $ ./preprocessing_train.sh
    $ ./preprocessing_test.sh
    $ ./train.sh
    $ ./test.sh

## 2. 使用说明

### 2.1 预处理

#### 2.1.1 预处理训练文件

训练文件的预处理包括:

 - 构建词表，即词->id的映射表，以及label表，以`dict`格式存放在`pkl`文件中;
 - 构建embedding表，根据所提供的预训练词向量文件，抽取所需要的向量，对于不存在于预训练文件中的词，则随机初始化。结果以`np.array`的格式存放在`pkl`文件中;
 - 将训练数据按顺序编号，每个实例写入单独的文件中，便于高效加载；
 - 统计句子长度，输出句子长度的[90, 95, 98, 100]百分位值;
 - 输出标签数量。

**运行方式:**

    $ python3 preprocessing.py -l --pd ./data/train.txt --ri ./data/train_idx/ --rv ./res/voc/

若需要使用预训练的词向量，则在上述命令之后添加下列命令，其中`path_to_embed_file`是预训练词向量路径，可以是`bin`或`txt`类型的文件:

    --re ./res/embed/ --pe ./path_to_embed_file

#### 2.1.2 预处理测试文件

**运行方式:**

    $ python3 preprocessing.py --pd ./data/test.txt --ri ./data/test_idx/

**表. 参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|l|bool|False|label，是否带有标签(标志是否是训练集)|
|pd|str|./data/train.txt|path_data，训练(测试)数据路径|
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|pe|str|None|path_embed，预训练的embed文件路径，`bin`或`txt`；若不提供，则随机初始化|
|pt|int|98|percentile，构建词表时的百分位值|

运行`python3 preprocessing.py -h`可打印出帮助信息。

### 2.2 训练

若预处理时`root_idx`等参数使用的是默认值，则在训练时不需要设定相应参数。

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0 python3 train.py --ml 90 --bs 256 -g

    # 使用第1列和第3列特征
    $ CUDA_VISIBLE_DEVICES=0 python3 train.py -f 0,2 --bs 256 --ml 90 -g

    # 使用第1列和第3列特征，并设置特征维度分别为64和16
    $ CUDA_VISIBLE_DEVICES=0 python3 train.py -f 0,2 --fd 64,16 --bs 256 --ml 90 -g

**参数说明**

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|f|list|[0]|features，训练时所使用的特征所在的列，若多列特征，则用逗号分隔|
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|ml|int|50|max_len，句子最大长度|
|ds|float|0.2|dev_size，开发集占比|
|lstm|int|256|lstm unit size，lstm单元数|
|ln|int|2|layer nums，lstm层数|
|fd|list|[64]|feature_dim，各列特征的维度，若多列特征，则用逗号分隔|
|dp|float|0.5|dropout_rate，dropout rate|
|lr|float|0.002|learning_rate，learning rate|
|ne|int|100|nb_epoch，迭代次数|
|mp|int|5|max_patience，最大耐心值，即开发集上性能超过mp次没有提示，则终止训练|
|rm|str|./model/|root_model，模型根目录|
|bs|int|64|batch_size，batch size|
|g|bool|False|是否使用GPU加速|
|nw|int|8|num_worker，加载数据时的线程数|

运行`python3 train.py -h`可打印出帮助信息。

### 2.3 测试

**运行方式:**

    $ CUDA_VISIBLE_DEVICES=0 python3 test.py --bs 256 -g -o ./result.txt

|参数|类型|默认值|说明|
| ------------ | ------------ | ------------ | ------------ |
|ri|str|./data/train_idx/|root_idx，训练数据索引文件根目录|
|rv|str|./res/voc/|root_voc，词表、label表根目录|
|re|str|./res/embed/|root_embed，embed文件根目录|
|pm|str|无|path_model，模型路径|
|bs|int|64|batch_size，batch size|
|g|bool|False|是否使用GPU加速|
|nw|int|8|num_worker，加载数据时的线程数|
|o|str|./result.txt|预测结果存放路径|

运行`python3 test.py -h`可打印出帮助信息。

## 3. Requirements

 - gensim==2.3.0
 - numpy==1.13.1
 - torch==0.2.0.post3
 - torchvision==0.1.9

## 4. 参考

 - [http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html](http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html "http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html")
 - [https://github.com/jiesutd/PyTorchSeqLabel](https://github.com/jiesutd/PyTorchSeqLabel "https://github.com/jiesutd/PyTorchSeqLabel")
 - [http://www.aclweb.org/anthology/N16-1030](http://www.aclweb.org/anthology/N16-1030 "http://www.aclweb.org/anthology/N16-1030")

## Updating

 - 2018-02-27: `bilstm.py`和`bilstm_crf.py`移入`TorchNN/layers`模块中，训练时使用Bi-LSTM-CRF模型.
 - 2018-02-26: `utils/`移入`TorchNN/`中；添加`layers`模块，并完成CRF层(未加入模型).
