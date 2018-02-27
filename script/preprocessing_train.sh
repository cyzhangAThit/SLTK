#Options:
#  -h, --help       show this help message and exit
#  -l, --label      数据是否带有标签(标志是否是训练集)
#  -f FEATURES      使用的特征列数
#  --pd=PATH_DATA   语料路径
#  --ri=ROOT_IDX    数据索引根目录
#  --rv=ROOT_VOC    字典根目录
#  --re=ROOT_EMBED  embed根目录
#  --pe=PATH_EMBED  embed文件路径
#  --pt=PT          构建word voc的百分位值


## 使用预训练的词向量
#python3 ../preprocessing.py \
#    -l \
#    --pd ../data/train.txt \
#    --ri ../data/train_idx \
#    --rv ../res/voc/ \
#    --pe ./CH.Gigaword.300B.300d.txt \
#    --re ../res/embed \
#    --pt 100


python3 ../preprocessing.py \
    -l \
    --pd ../data/train.txt \
    --ri ../data/train_idx \
    --rv ../res/voc/ \
    --pt 100
