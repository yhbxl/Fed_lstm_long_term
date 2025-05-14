import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        """初始化数据处理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.scalers = {}
        self.feature_importance = {}
        
    def preprocess_data(self, data: np.ndarray, client_id: int) -> np.ndarray:
        """预处理数据
        
        Args:
            data: 原始数据
            client_id: 客户端ID
            
        Returns:
            processed_data: 预处理后的数据
        """
        # 1. 标准化
        if client_id not in self.scalers:
            self.scalers[client_id] = StandardScaler()
            processed_data = self.scalers[client_id].fit_transform(data)
        else:
            processed_data = self.scalers[client_id].transform(data)
            
        # 2. 处理缺失值
        processed_data = np.nan_to_num(processed_data, nan=0.0)
        
        # 3. 处理异常值
        processed_data = self._handle_outliers(processed_data)
        
        return processed_data
        
    def _handle_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """处理异常值
        
        Args:
            data: 输入数据
            threshold: 阈值
            
        Returns:
            processed_data: 处理后的数据
        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # 使用Z-score方法处理异常值
        z_scores = np.abs((data - mean) / std)
        mask = z_scores > threshold
        
        # 将异常值替换为阈值
        processed_data = data.copy()
        processed_data[mask] = mean + threshold * std * np.sign(data[mask] - mean)
        
        return processed_data
        
    def augment_data(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强
        
        Args:
            data: 输入数据
            labels: 标签
            
        Returns:
            augmented_data: 增强后的数据
            augmented_labels: 增强后的标签
        """
        augmented_data = []
        augmented_labels = []
        
        for i in range(len(data)):
            # 原始数据
            augmented_data.append(data[i])
            augmented_labels.append(labels[i])
            
            # 添加高斯噪声
            noise = np.random.normal(0, 0.1, data[i].shape)
            augmented_data.append(data[i] + noise)
            augmented_labels.append(labels[i])
            
            # 随机缩放
            scale = np.random.uniform(0.9, 1.1)
            augmented_data.append(data[i] * scale)
            augmented_labels.append(labels[i])
            
        return np.array(augmented_data), np.array(augmented_labels)
        
    def calculate_feature_importance(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """计算特征重要性
        
        Args:
            data: 输入数据
            labels: 标签
            
        Returns:
            importance: 特征重要性字典
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # 使用随机森林计算特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(data, labels)
        
        importance = {
            f'feature_{i}': imp for i, imp in enumerate(rf.feature_importances_)
        }
        
        return importance
        
    def get_feature_stats(self, data: np.ndarray) -> Dict[str, Any]:
        """获取特征统计信息
        
        Args:
            data: 输入数据
            
        Returns:
            stats: 特征统计信息
        """
        stats = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0),
            'skewness': self._calculate_skewness(data),
            'kurtosis': self._calculate_kurtosis(data)
        }
        
        return stats
        
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """计算偏度"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z = (data - mean) / std
        return np.mean(z ** 3, axis=0)
        
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """计算峰度"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z = (data - mean) / std
        return np.mean(z ** 4, axis=0) - 3 