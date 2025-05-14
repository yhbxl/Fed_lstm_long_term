# 模拟客户端资源波动与状态历史

import random
import numpy as np
from torchvision import datasets
from typing import Dict, Any

class SimulatedClient:
    def __init__(self, cid):
        self.cid = cid
        self.cpu = random.uniform(0.2, 1.0)
        self.bandwidth = random.uniform(0.5, 1.5)
        self.energy = random.uniform(0.3, 1.0)
        self.history = []

    def simulate_step(self):
        self.cpu = max(0.1, self.cpu + np.random.normal(0, 0.05))
        self.bandwidth = max(0.1, self.bandwidth + np.random.normal(0, 0.05))
        self.energy = max(0.1, self.energy + np.random.normal(0, 0.05))

    def get_feature_vector(self):
        return [self.cpu, self.bandwidth, self.energy]

def create_federated_dataset(dataset_name: str, num_clients: int, non_iid_ratio: float = 0.5):
    """创建联邦学习数据集
    
    Args:
        dataset_name: 数据集名称
        num_clients: 客户端数量
        non_iid_ratio: 非独立同分布比例
    """
    if dataset_name == 'mnist':
        # 加载MNIST数据
        train_data = datasets.MNIST('./data', train=True, download=True)
        test_data = datasets.MNIST('./data', train=False, download=True)
        
        # 转换为numpy数组
        X_train = train_data.data.numpy()
        y_train = train_data.targets.numpy()
        X_test = test_data.data.numpy()
        y_test = test_data.targets.numpy()
        
    elif dataset_name == 'cifar10':
        # 加载CIFAR-10数据
        train_data = datasets.CIFAR10('./data', train=True, download=True)
        test_data = datasets.CIFAR10('./data', train=False, download=True)
        
        X_train = train_data.data
        y_train = np.array(train_data.targets)
        X_test = test_data.data
        y_test = np.array(test_data.targets)
        
    # 数据预处理
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    # 划分数据
    client_data = split_data_by_client(
        X_train, y_train,
        num_clients=num_clients,
        non_iid_ratio=non_iid_ratio
    )
    
    return client_data, (X_test, y_test)

def preprocess_dataset(data: np.ndarray, labels: np.ndarray, config: Dict[str, Any]):
    """预处理数据集
    
    Args:
        data: 输入数据
        labels: 标签
        config: 配置字典
    """
    processor = DataProcessor(config)
    
    # 1. 数据清洗
    processed_data = processor.preprocess_data(data, client_id=0)
    
    # 2. 数据增强
    if config['data']['augmentation']:
        processed_data, labels = processor.augment_data(processed_data, labels)
        
    # 3. 特征选择
    if config['data']['feature_selection']:
        importance = processor.calculate_feature_importance(processed_data, labels)
        selected_features = [f for f, imp in importance.items() if imp > 0.01]
        processed_data = processed_data[:, selected_features]
        
    return processed_data, labels
