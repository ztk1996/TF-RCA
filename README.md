# TraceStreama

## 快速开始

### 准备预处理数据

使用DataPreprocess目录中的SpanProcess.py来对原始数据集进行处理，原始数据需要以csv文件的形式作为输入，可以通过修改preprocess.py中的load_data函数来适配不同格式的数据，经过预处理后会以图的形式存储符合要求的完整trace在data.json文件中

预处理结果会存放在 `DataPreprocess/<数据集来源>/<预处理时间>` 目录下

#### 预处理正常数据集

1. 配置params.py

1. 运行命令

```shell
python SpanProcess.py --wechat
```

1. 保存最后生成的路径

#### 预处理带检测数据集

1. 配置params.py

1. 运行命令

```shell
python SpanProcess.py --wechat
```

1. 保存最后生成的路径

### 准备流处理数据

在分别完成完成了正常数据集与带检测数据集的预处理后，会在 `DataPreprocess/data` 目录下生成对应的data.json文件，TraceStream会直接从data.json中读取正常与异常数据并模拟流处理异常检测的过程

#### 正常数据集

正常数据集用以初始化最初的聚类微簇，需要尽可能覆盖系统中所有类型的trace：

1. 配置预处理后正常数据集路径

#### 配置检测数据集

1. 配置预处理后数据集路径
1. 配置数据集开始时间

### 调整异常检测参数

1. 配置时间窗口
