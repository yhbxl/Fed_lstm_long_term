# 基于LSTM的联邦学习客户端选择系统

本项目实现了一个基于LSTM的联邦学习客户端选择系统，通过预测客户端性能和资源状态来优化客户端选择策略，提高联邦学习的效率和稳定性。

## 项目特点

- 基于LSTM的客户端性能预测
- 动态客户端选择策略
- 数据漂移检测和处理
- 带宽预测和资源管理
- 完整的实验分析框架

## 系统架构

```
core/
├── bandwidth_predictor/     # 带宽预测模块
│   └── lstm_predictor.py    # LSTM带宽预测器
├── client_selector/         # 客户端选择模块
├── data/                    # 数据处理模块
│   ├── data_loader.py      # 数据加载器
│   ├── data_processor.py   # 数据处理器
│   └── data_utils.py       # 数据工具函数
├── fl_trainer/             # 联邦学习训练器
├── performance_predictor/   # 性能预测模块
│   ├── client_performance_predictor.py  # 客户端性能预测器
│   └── feature_extractor.py            # 特征提取器
├── data_drift_predictor/    # 数据漂移预测模块
└── utils/                  # 工具函数

experiments/
├── analysis/               # 实验分析工具
│   └── metrics_analyzer.py # 指标分析器
├── base/                   # 基础实验类
│   └── base_experiment.py  # 基础实验类
├── configs/               # 实验配置文件
└── runners/               # 实验运行器

datasets/                  # 数据集和模拟器
├── client_simulator.py    # 客户端模拟器
└── data_drift_simulator.py # 数据漂移模拟器
```

## 主要功能

1. **客户端性能预测**
   - 使用LSTM模型预测客户端性能
   - 特征提取和分析
   - 性能稳定性评估
   - 多维度指标分析

2. **动态客户端选择**
   - 基于多维度指标的选择策略
   - 资源状态感知
   - 自适应选择机制
   - 性能预测集成

3. **数据漂移处理**
   - 漂移检测
   - 自适应调整
   - 模型更新策略
   - 数据质量评估

4. **带宽预测**
   - LSTM-based带宽预测
   - 资源状态监控
   - 网络条件评估

5. **实验分析**
   - 性能指标分析
   - 可视化工具
   - 实验报告生成
   - 多维度评估

## 安装说明

1. 克隆项目：
```bash
git clone https://github.com/yourusername/fed_lstm_lastest.git
cd fed_lstm_long_term
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置实验参数：
```yaml
# experiments/configs/lstm_config.yaml
data:
  num_classes: 10
  clients: [1, 2, 3, 4, 5]
  batch_size: 32
  num_workers: 4
  augmentation: true
  feature_selection: true

model:
  name: "lstm"
  input_size: 64
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  num_rounds: 100
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 1e-4
```

2. 运行实验：
```python
from experiments.runners.lstm_runner import LSTMExperiment

# 初始化实验
experiment = LSTMExperiment(
    config_path="experiments/configs/lstm_config.yaml",
    output_dir="results"
)

# 运行实验
experiment.run()
```

## 数据集支持

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- EMNIST
- 自定义数据集

## 实验结果

实验结果将保存在`results`目录下，包括：
- 训练指标
- 性能分析
- 可视化图表
- 实验报告

## 依赖要求

- Python >= 3.7
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- numpy >= 1.19.5
- pandas >= 1.3.0
- scikit-learn >= 0.24.2
- matplotlib >= 3.4.2
- PyYAML >= 5.4.1
- tqdm >= 4.61.2

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 作者：Your Name
- 邮箱：your.email@example.com
- 项目地址：https://github.com/yourusername/fed_lstm_lastest
