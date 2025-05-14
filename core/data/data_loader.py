import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import datasets
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split

class FederatedDataset(Dataset):
    def __init__(self, data: np.ndarray, labels: np.ndarray, transform=None):
        """初始化联邦学习数据集
        
        Args:
            data: 数据数组
            labels: 标签数组
            transform: 数据转换函数
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

class FederatedDataLoader:
    def __init__(self, config: Dict[str, Any]):
        """初始化联邦学习数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.dataset_name = config['data']['dataset']
        self.num_clients = len(config['data']['clients'])
        self.batch_size = config['data']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.train_loaders = {}
        self.test_loaders = {}
        
    def load_data(self) -> Tuple[Dict[int, DataLoader], DataLoader]:
        """加载数据集并创建数据加载器
        
        Returns:
            Tuple[Dict[int, DataLoader], DataLoader]: 训练数据加载器和测试数据加载器
        """
        # 获取数据路径配置，如果不存在则使用默认路径
        data_path = self.config['data'].get('data_path', './data')
        
        if self.dataset_name == 'mnist':
            train_data = datasets.MNIST(data_path, train=True, download=True)
            test_data = datasets.MNIST(data_path, train=False, download=True)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == 'fashion_mnist':
            train_data = datasets.FashionMNIST(data_path, train=True, download=True)
            test_data = datasets.FashionMNIST(data_path, train=False, download=True)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        elif self.dataset_name == 'cifar10':
            train_data = datasets.CIFAR10(data_path, train=True, download=True)
            test_data = datasets.CIFAR10(data_path, train=False, download=True)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        # 转换为numpy数组
        X_train = train_data.data.numpy() if hasattr(train_data.data, 'numpy') else train_data.data
        y_train = train_data.targets.numpy() if hasattr(train_data.targets, 'numpy') else np.array(train_data.targets)
        X_test = test_data.data.numpy() if hasattr(test_data.data, 'numpy') else test_data.data
        y_test = test_data.targets.numpy() if hasattr(test_data.targets, 'numpy') else np.array(test_data.targets)
        
        # 数据预处理
        X_train = X_train.astype(np.float32) / 255.0
        X_test = X_test.astype(np.float32) / 255.0
        
        # 划分数据
        client_data = self._split_data_by_client(X_train, y_train)
        
        # 创建数据加载器
        train_loaders = {}
        for client_id, (X, y) in client_data.items():
            dataset = FederatedDataset(X, y)
            train_loaders[client_id] = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            
        test_dataset = FederatedDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        return train_loaders, test_loader
    
    def _split_data_by_client(self, X: np.ndarray, y: np.ndarray) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """将数据划分给不同的客户端
        
        Args:
            X: 特征数据
            y: 标签数据
            
        Returns:
            Dict[int, Tuple[np.ndarray, np.ndarray]]: 客户端ID到(特征,标签)的映射
        """
        client_data = {}
        non_iid_ratio = self.config['data'].get('non_iid_ratio', 0.5)
        
        if non_iid_ratio > 0:
            # 非独立同分布划分
            for i in range(self.num_clients):
                # 为每个客户端选择主要类别
                main_class = i % self.config['data']['num_classes']
                # 获取该类的所有样本
                class_indices = np.where(y == main_class)[0]
                # 随机选择样本
                selected_indices = np.random.choice(
                    class_indices,
                    size=len(class_indices) // self.num_clients,
                    replace=False
                )
                client_data[i] = (X[selected_indices], y[selected_indices])
        else:
            # 独立同分布划分
            indices = np.random.permutation(len(X))
            samples_per_client = len(X) // self.num_clients
            for i in range(self.num_clients):
                start_idx = i * samples_per_client
                end_idx = start_idx + samples_per_client
                client_indices = indices[start_idx:end_idx]
                client_data[i] = (X[client_indices], y[client_indices])
                
        return client_data
    
    def get_client_loaders(self, client_id: int) -> Tuple[DataLoader, DataLoader]:
        """获取客户端的数据加载器
        
        Args:
            client_id: 客户端ID
            
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        if client_id not in self.train_loaders:
            train_loader, test_loader = self.load_data()
            self.train_loaders[client_id] = train_loader[client_id]
            self.test_loaders[client_id] = test_loader
            
        return self.train_loaders[client_id], self.test_loaders[client_id]
        
    def get_data_stats(self, client_id: int, data: np.ndarray) -> Dict[str, Any]:
        """获取数据统计信息
        
        Args:
            client_id: 客户端ID
            data: 数据数组
            
        Returns:
            Dict: 数据统计信息
        """
        return {
            'num_samples': len(data),
            'num_features': data.shape[1] if len(data.shape) > 1 else 1,
            'feature_stats': {
                'mean': np.mean(data),
                'std': np.std(data),
                'min': np.min(data),
                'max': np.max(data)
            }
        }

def get_data_loaders(
    dataset: str, 
    batch_size: int, 
    num_clients: int,
    data_dir: str = 'datasets'  # 新增参数：数据集保存目录
) -> Tuple[List[DataLoader], DataLoader]:
    """获取训练和测试数据加载器
    
    Args:
        dataset: 数据集名称，支持 'mnist', 'cifar10', 'femnist'
        batch_size: 批次大小
        num_clients: 客户端数量
        data_dir: 数据集保存目录，默认为 'datasets'
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: (训练数据加载器列表, 测试数据加载器)
    """
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset.lower() == 'mnist':
        return get_mnist_loaders(batch_size, num_clients, data_dir)
    elif dataset.lower() == 'cifar10':
        return get_cifar10_loaders(batch_size, num_clients, data_dir)
    elif dataset.lower() == 'femnist':
        return get_femnist_loaders(batch_size, num_clients, data_dir)
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

def get_mnist_loaders(
    batch_size: int, 
    num_clients: int,
    data_dir: str = 'datasets'
) -> Tuple[List[DataLoader], DataLoader]:
    """获取MNIST数据集的数据加载器
    
    Args:
        batch_size: 批次大小
        num_clients: 客户端数量
        data_dir: 数据集保存目录
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: (训练数据加载器列表, 测试数据加载器)
    """
    # 创建MNIST专用目录
    mnist_dir = os.path.join(data_dir, 'mnist')
    os.makedirs(mnist_dir, exist_ok=True)
    
    # 定义变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.MNIST(
        root=mnist_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=mnist_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 分割训练数据
    return split_dataset(train_dataset, test_dataset, batch_size, num_clients)

def get_cifar10_loaders(
    batch_size: int, 
    num_clients: int,
    data_dir: str = 'datasets'
) -> Tuple[List[DataLoader], DataLoader]:
    """获取CIFAR10数据集的数据加载器
    
    Args:
        batch_size: 批次大小
        num_clients: 客户端数量
        data_dir: 数据集保存目录
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: (训练数据加载器列表, 测试数据加载器)
    """
    # 创建CIFAR10专用目录
    cifar10_dir = os.path.join(data_dir, 'cifar10')
    os.makedirs(cifar10_dir, exist_ok=True)
    
    # 定义变换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_dir, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar10_dir, 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # 分割训练数据
    return split_dataset(train_dataset, test_dataset, batch_size, num_clients)

def get_femnist_loaders(
    batch_size: int, 
    num_clients: int,
    data_dir: str = 'datasets'
) -> Tuple[List[DataLoader], DataLoader]:
    """获取FEMNIST数据集的数据加载器（简单模拟）
    
    Args:
        batch_size: 批次大小
        num_clients: 客户端数量
        data_dir: 数据集保存目录
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: (训练数据加载器列表, 测试数据加载器)
    """
    # 因为FEMNIST数据集不在torchvision中，这里我们用MNIST代替
    # 实际使用中应该替换为真实的FEMNIST数据集
    return get_mnist_loaders(batch_size, num_clients, data_dir)

def split_dataset(
    train_dataset: Dataset, 
    test_dataset: Dataset, 
    batch_size: int, 
    num_clients: int
) -> Tuple[List[DataLoader], DataLoader]:
    """将数据集分割给多个客户端
    
    Args:
        train_dataset: 训练数据集
        test_dataset: 测试数据集
        batch_size: 批次大小
        num_clients: 客户端数量
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: (训练数据加载器列表, 测试数据加载器)
    """
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    # 计算每个客户端的数据大小
    data_per_client = len(train_dataset) // num_clients
    
    # 创建训练数据加载器列表
    train_loaders = []
    
    # 为每个客户端分配数据
    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i < num_clients - 1 else len(train_dataset)
        
        # 创建子数据集
        indices = list(range(start_idx, end_idx))
        client_dataset = Subset(train_dataset, indices)
        
        # 创建数据加载器
        train_loader = DataLoader(
            client_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        train_loaders.append(train_loader)
    
    return train_loaders, test_loader 