# TraceStreama

## 快速开始

### 依赖环境

* python 3.9
* pandas
* numpy
* sklearn
* tqdm
* requests
* pymysql

### 准备预处理数据

使用 DataPreprocess 目录中的 SpanProcess.py 来对原始数据集进行处理，原始数据需要以 csv 文件的形式作为输入，可以通过修改 preprocess.py 中的 load_data 函数来适配不同格式的数据，经过预处理后会以图的形式存储符合要求的完整 trace 在 data.json 文件中

* 预处理结果会存放在 `DataPreprocess/<数据集来源>/<预处理时间>` 目录下
* 所有预处理数据集路径都是以 `data_root` 作为基本路径并安装数据来源进行划分
	eg. `mm_trace_root_list` 中配置项为 `data.csv` 的路径最终为 `data_root/wechat/data.csv`

#### 预处理初始化数据集

1. 配置路径：在 params.py 修改 `init_mm_data_path_list` 中的数据路径列表，对于微信数据，还需要在 `mm_trace_root_list` 中添加 clickstream 文件的路径

1. 运行命令

```shell
python SpanProcess.py --wechat
```

1. 保存预处理后数据的存放路径

#### 预处理带检测数据集

1. 配置路径：在 params.py 修改 `mm_data_path_list` 中的数据路径列表，对于微信数据，还需要在 `mm_trace_root_list` 中添加 clickstream 文件的路径

1. 运行命令

```shell
python SpanProcess.py --wechat
```

1. 保存预处理后数据的存放路径

### 准备流处理数据

在分别完成完成了正常数据集与带检测数据集的预处理后，会在 `DataPreprocess/data` 目录下生成对应的 data.json 文件，TraceStream会直接从 data.json 中读取正常与异常数据并模拟流处理异常检测的过程

#### 初始化数据集

正常数据集用以初始化最初的聚类微簇，初始化数据集需要尽可能覆盖系统中所有类型的 trace，并覆盖全部可能出现的 service：

1. 配置初始化数据集路径：在`init_Cluster`函数用于对 TraceStream 进行初始化操作，配置其中 open 操作打开的预处理后正常数据集路径，用于初始化的历史正常 trace 数据集。
1. 配置初始化数据集开始时间：`init_start_str` 用以初始化数据集的开始时间，即初始化数据集中第一条 trace 的时间戳（单位毫秒）。

#### 配置检测数据集

1. 配置预处理后数据集路径：在 main 函数中的 open 操作打开的是用于流式处理的 trace 数据集（数据集中可能存在异常 trace）
1. 配置数据集开始时间：`start_str` 是流式处理数据集的开始时间，即流式处理数据集中第一条 trace 的时间戳。
1. 配置各个case的根因信息：params.py中的`request_period_log`记录了各个故障 case 对应的根因和起始结束时间段。

### 参数调整

#### 全局参数设置

* window_duration_init：设置处理初始化数据集的时间窗口，即每次从初始化数据集中读取几分钟的 trace 数据，TraceStream 默认为 6 分钟。
* window_duration：设置处理流式处理数据集的时间窗口，即每次从流式处理数据集中读取几分钟的 trace 数据，TraceStream 默认为 6 分钟。
* rca_window：设置用于根因定位的单侧时间窗口大小。若检测到异常 trace，则取该 trace 前 rca_window 和后 rca_window 中的 trace 数据进行根因定位，TraceStream 默认为 3 分钟。

#### 异常检测模块参数设置

* eps：描述流式聚类算法中聚类半径的大小，TraceStream 中 eps 默认设置为 1，实际取值应根据数据分布而定。若 eps 设置过小，新到来的 trace 将更易被判为异常；若 eps 设置过大，新到来的 trace 更易被初始化产生的正常簇吸收，而被判为正常 trace。

#### 根因定位模块参数设置

* sRate：设置参与根因定位的 trace 的比例。sRate 最大设置为 1，表示时间窗口中的所有 trace 均参与根因定位；sRate 越小，表示参与根因定位的 trace 数越少。经实验验证，当 sRate 设置为 0.1，即仅有 10% 的 trace 参与根因定位时，根因定位的准确性无明显下降，但根因定位的效率明显提升。

### 运行

在配置好数据集路径以及调整好各参数后，通过以下方式运行：

```shell
python TraFlowRCA.py
```
