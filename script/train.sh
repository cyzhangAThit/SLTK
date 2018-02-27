#Options:
#    -h, --help         show this help message and exit
#    -f FEATURES        使用的特征列数
#    --ri=ROOT_IDX      数据索引根目录
#    --rv=ROOT_VOC      字典根目录
#    --re=ROOT_EMBED    embed根目录
#    --ml=MAX_LEN       实例最大长度
#    --ds=DEV_SIZE      开发集占比
#    --lstm=LSTM        LSTM单元数
#    --ln=LAYER_NUMS    LSTM层数
#    --fd=FEATURE_DIM   输入特征维度
#    --dp=DROPOUT       dropout rate
#    --lr=LEARN_RATE    learning rate
#    --ne=NB_EPOCH      迭代次数
#    --mp=MAX_PATIENCE  最大耐心值
#    --rm=ROOT_MODEL    模型根目录
#    --bs=BATCH_SIZE    batch size
#    -g, --cuda         是否使用GPU加速
#    --nw=NB_WORK       加载数据的线程数

## 使用预训练的词向量，加入`--re`参数
## 若使用预训练词向量，则`--fd`参数的第一个值，即词向量维度以预训练的为准
#CUDA_VISIBLE_DEVICES=0 python3 ../train.py \
#    -f 0 \
#    --ri ../data/train_idx \
#    --rv ../res/voc \
#    --ml 90 \
#    --fd 64 \
#    --re ../res/embed \
#    --ds 0.2 \
#    --lstm 256 \
#    --lr 0.002 \
#    --ne 1000 \
#    --mp 5 \
#    --bs 512 \
#    --rm ../model \
#    -g

CUDA_VISIBLE_DEVICES=0 python3 ../train.py \
    -f 0 \
    --ri ../data/train_idx \
    --rv ../res/voc \
    --ml 90 \
    --fd 64 \
    --ds 0.2 \
    --lstm 256 \
    --lr 0.002 \
    --ne 1000 \
    --mp 5 \
    --bs 512 \
    --rm ../model \
    -g
