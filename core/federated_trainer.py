# 联邦学习主训练流程

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from core.lstm_performance_predictor import PerformancePredictorManager

class FederatedTrainer:
    """联邦学习训练器，实现FedAvg算法"""
    
    def __init__(self, args, eval_fn):
        """初始化联邦学习训练器
        
        Args:
            args: 命令行参数对象
            eval_fn: 评估函数
        """
        self.args = args
        self.eval_fn = eval_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # 训练配置
        self.fraction_fit = args.selection_fraction
        self.min_clients = max(1, int(args.num_clients * args.selection_fraction))
        self.num_rounds = args.num_rounds
        
        # 初始化性能预测器
        predictor_config = {
            'window_size': args.lstm_sequence_length,
            'hidden_size': args.lstm_hidden_size,
            'num_layers': args.lstm_num_layers,
            'dropout': args.lstm_dropout,
            'learning_rate': args.lstm_learning_rate,
            'patience': args.lstm_patience
        }
        self.performance_predictor = PerformancePredictorManager(predictor_config)
        
        # 训练参数
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        
        # 指标记录
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'client_metrics': {}  # 客户端ID -> 指标历史
        }
        
    def train_client(self, client_id: int, model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader):
        """训练单个客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        # 训练模型
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        
        for epoch in range(1):
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                loss.backward()
                optimizer.step()
                
        # 评估客户端性能
        performance = self.evaluate_client(model, val_loader)
        
        # 更新性能预测器
        self.performance_predictor.update_client_history(client_id, performance)
        self.performance_predictor.train_predictor(client_id)
        
        return model, len(train_loader.dataset)
        
    def evaluate_client(self, model: nn.Module, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """评估客户端模型性能
        
        Args:
            model: 模型
            val_loader: 验证数据加载器
            
        Returns:
            性能指标字典
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        accuracy = correct / total
        avg_loss = total_loss / len(val_loader)
        
        return {'accuracy': accuracy, 'loss': avg_loss}
        
    def train(self, clients: List[Dict], initial_model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """执行联邦学习训练
        
        Args:
            clients: 客户端列表
            initial_model: 初始模型
            
        Returns:
            Tuple[nn.Module, Dict[str, Any]]: (训练后的模型, 训练指标)
        """
        current_model = initial_model
        best_model = None
        best_accuracy = 0.0
        
        for round_idx in range(self.num_rounds):
            print(f"\n=== 训练轮次 {round_idx + 1}/{self.num_rounds} ===")
            
            # 选择客户端
            num_selected = max(int(len(clients) * self.fraction_fit), self.min_clients)
            selected_clients = np.random.choice(clients, num_selected, replace=False)
            
            # 客户端训练
            client_weights = []
            client_models = []
            client_metrics = {}
            
            for client in selected_clients:
                client_id = client['id']
                
                # 训练客户端模型
                client_model, num_samples = self.train_client(
                    client_id=client_id,
                    model=current_model,
                    train_loader=client['train_loader'],
                    val_loader=client['val_loader']
                )
                
                # 评估客户端模型
                client_metric = self.evaluate_client(
                    model=client_model,
                    val_loader=client['val_loader']
                )
                
                # 记录客户端指标
                client_metrics[client_id] = client_metric
                
                # 保存训练结果
                client_models.append(client_model)
                client_weights.append(num_samples)
                
                # 保存客户端指标历史
                if client_id not in self.metrics_history['client_metrics']:
                    self.metrics_history['client_metrics'][client_id] = []
                self.metrics_history['client_metrics'][client_id].append(client_metric)
            
            # 根据样本数量计算权重
            total_samples = sum(client_weights)
            weights = [w / total_samples for w in client_weights]
            
            # 聚合模型（FedAvg）
            current_model = self.aggregate_models(client_models, weights)
            
            # 评估聚合后的全局模型
            val_loss, val_metrics = self.eval_fn(current_model)
            val_accuracy = val_metrics.get('accuracy', 0.0)
            
            # 更新指标历史
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_accuracy'].append(val_accuracy)
            
            print(f'轮次 {round_idx + 1}:')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  验证准确率: {val_accuracy:.4f}')
            
            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = current_model
        
        # 返回最佳模型和训练指标
        return best_model if best_model is not None else current_model, self.metrics_history
        
    def aggregate_models(self, models: List[nn.Module], weights: List[float]) -> nn.Module:
        """聚合模型（FedAvg）
        
        Args:
            models: 模型列表
            weights: 权重列表
            
        Returns:
            nn.Module: 聚合后的模型
        """
        # 创建新的模型实例
        aggregated_model = type(models[0])()
        
        # 聚合参数
        for param in aggregated_model.parameters():
            param.data.zero_()
            
        for model, weight in zip(models, weights):
            for param, param_aggregated in zip(model.parameters(), aggregated_model.parameters()):
                param_aggregated.data += param.data * weight
                
        return aggregated_model
        
    def get_client_predictions(self, client_ids: List[int]) -> Dict[int, float]:
        """获取客户端性能预测
        
        Args:
            client_ids: 客户端ID列表
            
        Returns:
            客户端ID到预测性能的映射
        """
        return self.performance_predictor.get_all_predictions(client_ids)

    def evaluate_global_model(self, model: nn.Module) -> Tuple[float, Dict[str, float]]:
        """评估全局模型性能
        
        Args:
            model: 全局模型
            
        Returns:
            Tuple[float, Dict[str, float]]: (损失值, 性能指标)
        """
        return self.eval_fn(model)


