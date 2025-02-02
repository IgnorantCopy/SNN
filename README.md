# 基于 STDBP 算法的脉冲神经网络 FPGA 加速实现与优化

## 1. 模型训练

### 1.1 ANN2SNN 方法

#### 1.1.1 ANN 的预训练

>  script: hybird 目录下的 `ann.py`

1. 运行方式

   + 终端（进入到 `ann.py` 所在目录）

   ```powershell
   python ann.py --dataset MNIST --batch_size 64 -lr 0.001 -e 100 --optimizer SGD
   ```

   + Pycharm

   ![1738483637765](README.assets/1738483637765.png)

   ![1738497711401](README.assets/1738497711401.png)

   随后点击运行即可

2. 参数说明

|          参数          |          描述          |   默认值    |
| :--------------------: | :--------------------: | :---------: |
|       --dataset        |       数据集名称       |    MNIST    |
|     --dataset_root     |    数据集下载根目录    | E:/DataSets |
|      --batch_size      | 超参数 batch_size 大小 |     64      |
| -lr 或 --learning_rate |      超参数学习率      |    1e-3     |
|     --weight_decay     |  超参数 weight_decay   |    5e-4     |
|    -e 或 --epoches     |     超参数学习轮数     |     100     |
|      --optimizer       |         优化器         |     SGD     |
|         --gpu          |   是否使用 GPU 加速    |    False    |
|         --log          | 是否将输出重定向为日志 |    True     |
|       --log_dir        |      日志保存目录      |   ./logs    |
|      --model_dir       |      模型保存目录      |  ./models   |
|   --pretrained_model   |     预训练模型位置     |    None     |



### 1.2 SNN from Scratch



### 1.3 Hybrid Trainning

