import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split

def load_data_from_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """从文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        data: 数据数组
        labels: 标签数组
    """
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
    elif file_path.endswith('.npy'):
        data = np.load(file_path)
        labels = np.load(file_path.replace('data', 'labels'))
    else:
        raise ValueError("不支持的文件格式")
        
    return data, labels

def save_data_to_file(data: np.ndarray, labels: np.ndarray, file_path: str):
    """保存数据到文件
    
    Args:
        data: 数据数组
        labels: 标签数组
        file_path: 文件路径
    """
    if file_path.endswith('.csv'):
        df = pd.DataFrame(data)
        df['label'] = labels
        df.to_csv(file_path, index=False)
    elif file_path.endswith('.npy'):
        np.save(file_path, data)
        np.save(file_path.replace('data', 'labels'), labels)
    else:
        raise ValueError("不支持的文件格式")

def split_data_by_client(
    data: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    non_iid_ratio: float = 0.5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """按客户端划分数据
    
    Args:
        data: 数据数组
        labels: 标签数组
        num_clients: 客户端数量
        non_iid_ratio: 非独立同分布比例
        
    Returns:
        client_data: 客户端数据列表
    """
    # 1. 按标签排序
    sorted_indices = np.argsort(labels)
    data = data[sorted_indices]
    labels = labels[sorted_indices]
    
    # 2. 计算每个客户端的样本数
    samples_per_client = len(data) // num_clients
    
    # 3. 划分数据
    client_data = []
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client
        
        # 非独立同分布：每个客户端主要包含部分类别的数据
        if non_iid_ratio > 0:
            num_classes = len(np.unique(labels))
            main_class = i % num_classes
            
            # 获取主要类别的样本
            main_class_indices = np.where(labels == main_class)[0]
            main_class_samples = int(samples_per_client * non_iid_ratio)
            
            # 获取其他类别的样本
            other_class_indices = np.where(labels != main_class)[0]
            other_class_samples = samples_per_client - main_class_samples
            
            # 随机选择样本
            main_class_selected = np.random.choice(
                main_class_indices,
                size=main_class_samples,
                replace=False
            )
            other_class_selected = np.random.choice(
                other_class_indices,
                size=other_class_samples,
                replace=False
            )
            
            selected_indices = np.concatenate([main_class_selected, other_class_selected])
            np.random.shuffle(selected_indices)
            
            client_data.append((data[selected_indices], labels[selected_indices]))
        else:
            # 独立同分布：随机划分
            indices = np.random.permutation(len(data))[start_idx:end_idx]
            client_data.append((data[indices], labels[indices]))
            
    return client_data

def generate_synthetic_data(
    num_samples: int,
    num_features: int,
    num_classes: int,
    noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成数据
    
    Args:
        num_samples: 样本数量
        num_features: 特征数量
        num_classes: 类别数量
        noise_level: 噪声水平
        
    Returns:
        data: 数据数组
        labels: 标签数组
    """
    # 1. 生成类别中心
    centers = np.random.randn(num_classes, num_features)
    
    # 2. 生成数据
    data = []
    labels = []
    
    samples_per_class = num_samples // num_classes
    for i in range(num_classes):
        # 生成该类别的样本
        class_data = centers[i] + noise_level * np.random.randn(samples_per_class, num_features)
        data.append(class_data)
        labels.append(np.ones(samples_per_class) * i)
        
    # 3. 合并数据
    data = np.vstack(data)
    labels = np.concatenate(labels)
    
    # 4. 打乱数据
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    return data, labels

def calculate_data_quality(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算数据质量指标
    
    Args:
        data: 数据数组
        labels: 标签数组
        
    Returns:
        quality_metrics: 质量指标字典
    """
    # 1. 计算类别平衡度
    class_counts = np.bincount(labels.astype(int))
    class_balance = np.min(class_counts) / np.max(class_counts)
    
    # 2. 计算特征相关性
    feature_corr = np.corrcoef(data.T)
    feature_corr = np.abs(feature_corr)
    np.fill_diagonal(feature_corr, 0)
    avg_correlation = np.mean(feature_corr)
    
    # 3. 计算数据密度
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=2).fit(data)
    distances, _ = nbrs.kneighbors(data)
    avg_distance = np.mean(distances[:, 1])
    
    # 4. 计算类别可分性
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    try:
        lda.fit(data, labels)
        separability = np.mean(lda.explained_variance_ratio_)
    except:
        separability = 0.0
        
    quality_metrics = {
        'class_balance': class_balance,
        'feature_correlation': avg_correlation,
        'data_density': 1.0 / (1.0 + avg_distance),
        'class_separability': separability
    }
    
    return quality_metrics 