import numpy as np
from typing import List, Dict, Any, Tuple
import torch
from collections import deque
import copy

class LongTermClientSelector:
    """基于论文的长期贪心客户端选择策略，结合LSTM预测和混合评分机制"""
    
    def __init__(self, args):
        """初始化长期客户端选择器
        
        Args:
            args: 命令行参数对象
        """
        self.args = args
        
        # 从args获取参数
        self.window_size = args.window_size
        self.diversity_weight = 1.0 - args.performance_weight
        self.performance_weight = args.performance_weight
        self.min_clients = max(1, int(args.num_clients * args.selection_fraction))
        self.max_clients = args.num_clients
        self.beta = 10.0  # 论文中的β放大因子
        
        # 客户端历史记录
        self.client_history = {}  # 客户端ID -> 历史性能记录
        self.client_states = {}   # 客户端ID -> 客户端状态
        self.client_weights = {}  # 客户端ID -> 权重
        
    def update_client_history(self, client_id: int, metrics: Dict[str, float]):
        """更新客户端历史记录
        
        Args:
            client_id: 客户端ID
            metrics: 性能指标，例如 {'accuracy': 0.85, 'loss': 0.1}
        """
        if client_id not in self.client_history:
            self.client_history[client_id] = deque(maxlen=self.window_size)
        
        self.client_history[client_id].append(metrics)
    
    def update_client_state(self, client_id: int, state: Dict[str, Any]):
        """更新客户端状态
        
        Args:
            client_id: 客户端ID
            state: 客户端状态信息，包含带宽、计算能力等
        """
        self.client_states[client_id] = state
    
    def calculate_score(self, client_id: int) -> float:
        """计算客户端综合评分，实现论文中的评分公式
        
        Args:
            client_id: 客户端ID
            
        Returns:
            float: 客户端评分
        """
        if client_id not in self.client_history or len(self.client_history[client_id]) == 0:
            return 0.5  # 对于没有历史记录的客户端，给予中等分数
        
        # 最新一轮的性能作为性能得分
        recent_metrics = self.client_history[client_id][-1]
        performance_score = recent_metrics.get('accuracy', 0.5)
        
        # 历史性能方差作为多样性得分
        if len(self.client_history[client_id]) >= 2:
            accuracy_history = [metrics.get('accuracy', 0.5) for metrics in self.client_history[client_id]]
            diversity_score = np.var(accuracy_history) * self.beta  # 使用β放大方差
        else:
            diversity_score = 0.1  # 历史记录不足时给予较小的多样性分数
        
        # 混合得分 = α * 性能得分 + (1-α) * 多样性得分
        score = (
            self.performance_weight * performance_score + 
            self.diversity_weight * diversity_score
        )
        
        return score
    
    def select_clients(self, 
                      available_clients: List[Any], 
                      num_selected: int,
                      lstm_predictor=None) -> List[Any]:
        """基于长期贪心策略选择客户端
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            lstm_predictor: LSTM性能预测器（可选）
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        # 确保选择数量在合理范围内
        num_selected = min(num_selected, len(available_clients))
        num_selected = max(num_selected, self.min_clients)
        num_selected = min(num_selected, self.max_clients)
        
        # 如果历史记录不足，使用随机选择
        if len(self.client_history) < self.min_clients:
            return self._random_selection(available_clients, num_selected)
        
        # 计算每个客户端的评分
        client_scores = []
        
        for client in available_clients:
            client_id = client['id']
            
            # 使用LSTM预测器获取预测性能（如果提供）
            if lstm_predictor is not None and client_id in self.client_history:
                predicted_score = lstm_predictor.predict_performance(client_id)
                if predicted_score is not None:
                    # 使用预测性能作为当前性能分数
                    if client_id not in self.client_history:
                        self.client_history[client_id] = deque(maxlen=self.window_size)
                    
                    # 构造预测指标
                    predicted_metrics = {'accuracy': predicted_score}
                    
                    # 临时添加预测指标计算评分
                    original_history = copy.deepcopy(self.client_history[client_id])
                    self.client_history[client_id].append(predicted_metrics)
                    score = self.calculate_score(client_id)
                    
                    # 恢复原始历史记录
                    self.client_history[client_id] = original_history
                else:
                    # 无法预测时使用当前评分
                    score = self.calculate_score(client_id)
            else:
                # 无预测器时使用当前评分
                score = self.calculate_score(client_id)
                
            client_scores.append((client, score))
            
        # 按评分降序排序
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择得分最高的客户端
        selected_clients = [client for client, _ in client_scores[:num_selected]]
        
        return selected_clients
    
    def _random_selection(self, available_clients: List[Any], num_selected: int) -> List[Any]:
        """随机选择客户端（作为备选策略）
        
        Args:
            available_clients: 可用客户端列表
            num_selected: 需要选择的客户端数量
            
        Returns:
            List[Any]: 选中的客户端列表
        """
        import random
        # 确保选择数量不超过可用客户端数量
        num_selected = min(num_selected, len(available_clients))
        
        # 随机选择客户端
        selected_indices = random.sample(range(len(available_clients)), num_selected)
        return [available_clients[i] for i in selected_indices]
    
    def adjust_weights(self, client_id: int, performance_improvement: float):
        """根据性能提升调整客户端权重
        
        Args:
            client_id: 客户端ID
            performance_improvement: 性能提升值
        """
        if client_id not in self.client_weights:
            self.client_weights[client_id] = 1.0
            
        # 根据性能提升调整权重
        # 如果性能提升为正，增加权重；如果为负，减少权重
        weight_adjustment = 1.0 + performance_improvement
        
        # 确保权重在合理范围内
        self.client_weights[client_id] = max(0.1, min(2.0, weight_adjustment))
    
    def adjust_window_size(self, performance_improvement: float):
        """根据性能提升调整观察窗口大小
        
        Args:
            performance_improvement: 性能提升值
        """
        # 如果性能提升为正，增加窗口大小；如果为负，减少窗口大小
        if performance_improvement > self.args.window_adjust_threshold:
            # 性能提升显著，增加窗口大小以获取更多历史信息
            self.window_size = min(self.window_size + 1, self.args.max_window_size)
        elif performance_improvement < -self.args.window_adjust_threshold:
            # 性能下降，减少窗口大小以更快适应变化
            self.window_size = max(self.window_size - 1, self.args.min_window_size)
            
        # 更新所有客户端历史记录的最大长度
        for client_id in self.client_history:
            self.client_history[client_id] = deque(
                self.client_history[client_id],
                maxlen=self.window_size
            ) 