# 联邦学习主训练流程

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from core.client_selector.client_runner import train_on_client, evaluate_model, aggregate_models
from core.client_selector.client_selector import ClientSelector

class FederatedTrainer:
    def __init__(
        self,
        client_selector: ClientSelector,
        eval_fn: callable,
        config: Dict[str, Any],
        **kwargs
    ):
        """初始化联邦学习训练器
        
        Args:
            client_selector: 客户端选择器
            eval_fn: 评估函数
            config: 训练配置
        """
        self.client_selector = client_selector
        self.eval_fn = eval_fn
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
        # 设置随机种子
        if 'system' in config and 'seed' in config['system']:
            torch.manual_seed(config['system']['seed'])
            if torch.cuda.is_available():
                torch.cuda.manual_seed(config['system']['seed'])
                
        # 训练配置
        self.fraction_fit = config.get('client_selection', {}).get('fraction_fit', 0.5)
        self.min_clients = config.get('client_selection', {}).get('min_clients', 5)
        self.num_rounds = config.get('training', {}).get('num_rounds', 100)
        
    def train(self, clients: List[Any], initial_model: nn.Module) -> nn.Module:
        """执行联邦学习训练
        
        Args:
            clients: 客户端列表
            initial_model: 初始模型
            
        Returns:
            nn.Module: 训练后的模型
        """
        current_model = initial_model
        best_accuracy = 0.0
        
        for round_idx in range(self.num_rounds):
            print(f"\n=== 训练轮次 {round_idx + 1}/{self.num_rounds} ===")
            
            # 选择客户端
            num_selected = max(int(len(clients) * self.fraction_fit), self.min_clients)
            selected_clients = self.client_selector.select_clients(
                available_clients=clients,
                num_selected=num_selected
            )
            
            # 客户端训练
            client_weights = []
            client_models = []
            total_samples = 0
            
            for client in selected_clients:
                # 训练客户端模型
                client_model, num_samples = train_on_client(
                    client=client,
                    model=current_model,
                    config=self.config
                )
                
                client_models.append(client_model)
                client_weights.append(num_samples)
                total_samples += num_samples
                
            # 聚合模型
            weights = [w / total_samples for w in client_weights]
            current_model = aggregate_models(client_models, weights)
            
            # 评估模型
            loss, metrics = self.eval_fn(current_model)
            accuracy = metrics.get('accuracy', 0.0)
            
            print(f'轮次 {round_idx + 1}:')
            print(f'  损失: {loss:.4f}')
            print(f'  准确率: {accuracy:.4f}')
            
            # 保存最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = current_model
                
        return best_model


