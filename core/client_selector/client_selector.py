import numpy as np
import torch
from typing import List, Dict, Any, Tuple
import random
from collections import deque

class ClientSelector:
    """客户端选择器，实现不同的客户端选择策略"""
    
    def __init__(self, strategy: str, config: Dict[str, Any]):
        """初始化客户端选择器
        
        Args:
            strategy: 选择策略，可选值: "random", "lstm", "diversity", "performance", "hybrid"
            config: 配置信息
        """
        self.strategy = strategy
        self.config = config
        self.client_history = {}  # 保存每个客户端的历史性能
        self.diversity_weight = config.get('client_selection', {}).get('diversity_weight', 0.5)
        self.performance_weight = config.get('client_selection', {}).get('performance_weight', 0.5)
        self.window_size = config.get('client_selection', {}).get('window_size', 5)
        self.min_clients = config.get('client_selection', {}).get('min_clients', 3)
        self.max_clients = config.get('client_selection', {}).get('max_clients', 10)
        
    def select_clients(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """选择客户端
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        if self.strategy == "random":
            return self._random_selection(available_clients, num_selected)
        elif self.strategy == "diversity":
            return self._diversity_based_selection(available_clients, num_selected)
        elif self.strategy == "performance":
            return self._performance_based_selection(available_clients, num_selected)
        elif self.strategy == "hybrid":
            return self._hybrid_selection(available_clients, num_selected)
        else:
            # 默认使用随机选择
            return self._random_selection(available_clients, num_selected)
            
    def update_client_history(self, client_id: int, metrics: Dict[str, float]):
        """更新客户端历史记录
        
        Args:
            client_id: 客户端ID
            metrics: 性能指标，如 {'accuracy': 0.85, 'loss': 0.1, ...}
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = deque(maxlen=self.window_size)
        
        self.client_history[client_id].append(metrics)
            
    def _random_selection(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """随机选择客户端
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 确保选择数量不超过可用客户端数量
        num_selected = min(num_selected, len(available_clients))
        
        # 随机选择客户端
        selected_indices = random.sample(range(len(available_clients)), num_selected)
        return [available_clients[i] for i in selected_indices]
        
    def _diversity_based_selection(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """基于多样性的客户端选择
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 如果历史记录不足，使用随机选择
        if len(self.client_history) < self.min_clients:
            return self._random_selection(available_clients, num_selected)
        
        # 计算每个客户端的数据多样性得分
        diversity_scores = {}
        for client in available_clients:
            client_id = client['id']
            if client_id in self.client_history and len(self.client_history[client_id]) > 0:
                # 这里使用简单的方法估计数据多样性：历史准确率的方差
                accuracy_history = [metrics.get('accuracy', 0.5) for metrics in self.client_history[client_id]]
                diversity_scores[client_id] = np.var(accuracy_history) * 10  # 放大方差使其更有影响
            else:
                # 对于没有历史记录的客户端，给予中等多样性得分
                diversity_scores[client_id] = 0.5
        
        # 根据多样性得分选择客户端
        client_scores = [(client, diversity_scores.get(client['id'], 0.5)) for client in available_clients]
        client_scores.sort(key=lambda x: x[1], reverse=True)  # 按多样性得分降序排序
        
        return [client for client, _ in client_scores[:num_selected]]
        
    def _performance_based_selection(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """基于性能的客户端选择
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 如果历史记录不足，使用随机选择
        if len(self.client_history) < self.min_clients:
            return self._random_selection(available_clients, num_selected)
        
        # 计算每个客户端的性能得分
        performance_scores = {}
        for client in available_clients:
            client_id = client['id']
            if client_id in self.client_history and len(self.client_history[client_id]) > 0:
                # 使用最近的准确率作为性能得分
                recent_metrics = self.client_history[client_id][-1]
                performance_scores[client_id] = recent_metrics.get('accuracy', 0.5)
            else:
                # 对于没有历史记录的客户端，给予中等性能得分
                performance_scores[client_id] = 0.5
        
        # 根据性能得分选择客户端
        client_scores = [(client, performance_scores.get(client['id'], 0.5)) for client in available_clients]
        client_scores.sort(key=lambda x: x[1], reverse=True)  # 按性能得分降序排序
        
        return [client for client, _ in client_scores[:num_selected]]
        
    def _hybrid_selection(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """结合多样性和性能的混合选择策略
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 如果历史记录不足，使用随机选择
        if len(self.client_history) < self.min_clients:
            return self._random_selection(available_clients, num_selected)
        
        # 计算每个客户端的混合得分
        hybrid_scores = {}
        for client in available_clients:
            client_id = client['id']
            if client_id in self.client_history and len(self.client_history[client_id]) > 0:
                # 计算性能得分：最近一次训练的准确率
                recent_metrics = self.client_history[client_id][-1]
                performance_score = recent_metrics.get('accuracy', 0.5)
                
                # 计算多样性得分：历史准确率的方差
                accuracy_history = [metrics.get('accuracy', 0.5) for metrics in self.client_history[client_id]]
                diversity_score = np.var(accuracy_history) * 10  # 放大方差使其更有影响
                
                # 计算混合得分
                hybrid_scores[client_id] = (
                    self.performance_weight * performance_score + 
                    self.diversity_weight * diversity_score
                )
            else:
                # 对于没有历史记录的客户端，给予中等混合得分
                hybrid_scores[client_id] = 0.5
        
        # 根据混合得分选择客户端
        client_scores = [(client, hybrid_scores.get(client['id'], 0.5)) for client in available_clients]
        client_scores.sort(key=lambda x: x[1], reverse=True)  # 按混合得分降序排序
        
        return [client for client, _ in client_scores[:num_selected]] 