import torch
import torch.nn as nn
import numpy as np
from collections import deque
import flwr as fl

class ModelFeatureExtractor:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        
    def extract_features(self, data_loader, model, device):
        """提取模型长期表现相关的特征"""
        features = []
        model.eval()
        
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                
                # 1. 模型预测特征
                outputs = model(batch_data)
                pred_probs = torch.softmax(outputs, dim=1)
                
                # 预测熵（反映模型的不确定性）
                pred_entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-10), dim=1)
                
                # 预测置信度
                pred_confidence = torch.max(pred_probs, dim=1)[0]
                
                # 2. 数据分布特征
                mean = torch.mean(batch_data, dim=0)
                std = torch.std(batch_data, dim=0)
                
                # 3. 类别分布特征
                unique_labels, counts = torch.unique(batch_labels, return_counts=True)
                class_dist = torch.zeros(self.num_classes).to(device)
                class_dist[unique_labels] = counts.float() / len(batch_labels)
                
                # 4. 模型性能特征
                correct = (torch.argmax(outputs, dim=1) == batch_labels).float()
                accuracy = torch.mean(correct)
                
                # 5. 数据质量特征
                data_quality = torch.mean(pred_confidence)  # 模型对数据的置信度
                data_diversity = torch.mean(pred_entropy)   # 预测分布的多样性
                
                # 6. 长期表现相关特征
                # 计算每个类别的平均置信度
                class_confidence = torch.zeros(self.num_classes).to(device)
                for i in range(self.num_classes):
                    mask = (batch_labels == i)
                    if mask.any():
                        class_confidence[i] = torch.mean(pred_confidence[mask])
                
                # 计算类别平衡度
                class_balance = 1.0 - torch.std(class_dist)
                
                # 7. 新增：模型稳定性特征
                # 计算预测分布的方差
                pred_variance = torch.var(pred_probs, dim=1)
                pred_variance_mean = torch.mean(pred_variance)
                
                # 计算预测分布的偏度
                pred_skewness = torch.mean(((pred_probs - pred_probs.mean(dim=1, keepdim=True)) / 
                                          (pred_probs.std(dim=1, keepdim=True) + 1e-10)) ** 3, dim=1)
                pred_skewness_mean = torch.mean(pred_skewness)
                
                # 8. 新增：数据复杂度特征
                # 计算数据点的L2范数
                data_norm = torch.norm(batch_data, dim=1)
                data_norm_mean = torch.mean(data_norm)
                data_norm_std = torch.std(data_norm)
                
                # 9. 新增：模型鲁棒性特征
                # 计算预测概率的梯度
                batch_data.requires_grad_(True)
                outputs = model(batch_data)
                pred_probs = torch.softmax(outputs, dim=1)
                pred_probs.backward(torch.ones_like(pred_probs))
                gradients = batch_data.grad
                gradient_norm = torch.norm(gradients, dim=1)
                gradient_norm_mean = torch.mean(gradient_norm)
                
                # 组合所有特征
                batch_features = torch.cat([
                    # 模型预测特征
                    torch.mean(pred_entropy).unsqueeze(0),
                    torch.mean(pred_confidence).unsqueeze(0),
                    
                    # 数据分布特征
                    mean, std,
                    
                    # 类别分布特征
                    class_dist,
                    
                    # 模型性能特征
                    accuracy.unsqueeze(0),
                    
                    # 数据质量特征
                    data_quality.unsqueeze(0),
                    data_diversity.unsqueeze(0),
                    
                    # 长期表现特征
                    class_confidence,
                    class_balance.unsqueeze(0),
                    
                    # 模型稳定性特征
                    pred_variance_mean.unsqueeze(0),
                    pred_skewness_mean.unsqueeze(0),
                    
                    # 数据复杂度特征
                    data_norm_mean.unsqueeze(0),
                    data_norm_std.unsqueeze(0),
                    
                    # 模型鲁棒性特征
                    gradient_norm_mean.unsqueeze(0)
                ])
                
                features.append(batch_features)
                
        return torch.stack(features)
    
    def calculate_feature_importance(self, features, performance_metrics):
        """计算特征重要性"""
        # 使用简单的相关性分析
        correlations = []
        for i in range(features.shape[1]):
            correlation = np.corrcoef(features[:, i].cpu().numpy(), 
                                    performance_metrics.cpu().numpy())[0, 1]
            correlations.append(abs(correlation))
        return torch.tensor(correlations)
    
    def get_feature_names(self):
        """返回特征名称列表"""
        return [
            'prediction_entropy',
            'prediction_confidence',
            'data_mean',
            'data_std',
            'class_distribution',
            'accuracy',
            'data_quality',
            'data_diversity',
            'class_confidence',
            'class_balance',
            'prediction_variance',
            'prediction_skewness',
            'data_norm_mean',
            'data_norm_std',
            'gradient_norm_mean'
        ]

class EnhancedClientSelector:
    def __init__(self, num_clients, feature_size, min_window_size=3, max_window_size=10):
        self.num_clients = num_clients
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.current_window_size = min_window_size
        
        # 集成多个分析器
        self.performance_analyzer = ClientPerformanceAnalyzer(feature_size)
        self.drift_analyzer = DataDriftAnalyzer(feature_size)
        self.bandwidth_predictor = BandwidthPredictorManager(num_clients)
        
        # 客户端性能记录
        self.client_metrics = {
            'performance': {},    # 性能指标
            'stability': {},      # 稳定性指标
            'bandwidth': {},      # 带宽记录
            'data_quality': {},   # 数据质量
            'training_time': {},  # 训练时间
            'last_selected': {},  # 上次选中时间
            'contribution': {}    # 贡献度
        }
        
        # 观测窗口
        self.observation_window = deque(maxlen=max_window_size)
        self.current_round = 0
        
    def update_metrics(self, client_id, data_loader, model, device, training_time):
        """更新客户端综合指标"""
        # 更新性能分析
        performance_metrics = self.performance_analyzer.get_client_metrics(client_id)
        self.client_metrics['performance'][client_id] = performance_metrics
        
        # 更新数据漂移分析
        drift_metrics = self.drift_analyzer.get_client_metrics(client_id)
        self.client_metrics['data_quality'][client_id] = drift_metrics['data_quality']
        
        # 更新带宽预测
        bandwidth = self.bandwidth_predictor.predict_bandwidth(client_id)
        if bandwidth is not None:
            self.client_metrics['bandwidth'][client_id] = bandwidth
            
        # 更新训练时间
        self.client_metrics['training_time'][client_id] = training_time
        self.client_metrics['last_selected'][client_id] = self.current_round
        
    def calculate_client_score(self, client_id):
        """计算客户端综合得分"""
        if client_id not in self.client_metrics['performance']:
            return float('-inf')
            
        # 获取各项指标
        performance = self.client_metrics['performance'][client_id]
        data_quality = self.client_metrics['data_quality'].get(client_id, 0.0)
        bandwidth = self.client_metrics['bandwidth'].get(client_id, 0.0)
        training_time = self.client_metrics['training_time'].get(client_id, float('inf'))
        
        # 计算各项得分
        performance_score = performance['overall_score']
        stability_score = performance['stability_score']
        time_efficiency = 1.0 / (1.0 + training_time)
        bandwidth_score = 1.0 / (1.0 + 1.0/bandwidth) if bandwidth > 0 else 0.0
        
        # 计算综合得分
        score = (
            0.35 * performance_score +    # 性能得分
            0.25 * stability_score +      # 稳定性得分
            0.20 * data_quality +         # 数据质量得分
            0.10 * time_efficiency +      # 时间效率得分
            0.10 * bandwidth_score        # 带宽得分
        )
        
        return score
        
    def select_clients(self, num_selected):
        """选择客户端"""
        self.current_round += 1
        
        # 更新观测窗口
        self.observation_window.append(self.current_round)
        
        # 如果观测窗口未满，使用探索策略
        if len(self.observation_window) < self.current_window_size:
            return self._exploration_selection(num_selected)
            
        # 计算所有客户端的得分
        client_scores = {}
        for client_id in range(self.num_clients):
            client_scores[client_id] = self.calculate_client_score(client_id)
            
        # 使用softmax进行概率采样
        scores = np.array(list(client_scores.values()))
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # 选择客户端
        selected_clients = np.random.choice(
            self.num_clients,
            size=min(num_selected, self.num_clients),
            replace=False,
            p=probs
        )
        
        return selected_clients.tolist()
        
    def _exploration_selection(self, num_selected):
        """探索策略选择客户端"""
        # 优先选择未充分探索的客户端
        unexplored = set(range(self.num_clients)) - set(self.client_metrics['last_selected'].keys())
        if unexplored:
            return list(np.random.choice(list(unexplored), 
                                       size=min(num_selected, len(unexplored)),
                                       replace=False))
                                       
        # 否则随机选择
        return np.random.choice(self.num_clients, 
                              size=min(num_selected, self.num_clients),
                              replace=False).tolist()

class EnhancedFederatedTrainer(fl.server.strategy.FedAvg):
    def __init__(self, client_selector, eval_fn, **kwargs):
        super().__init__(**kwargs)
        self.client_selector = client_selector
        self.eval_fn = eval_fn
        self.round_metrics = []
        
    def configure_fit(self, server_round, parameters, client_manager):
        """配置客户端训练"""
        # 选择客户端
        selected_clients = self.client_selector.select_clients(
            num_selected=self.min_fit_clients
        )
        
        # 准备配置
        config = {
            "server_round": server_round,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }
        
        return [(client_manager.clients[cid], parameters, config) 
                for cid in selected_clients]
                
    def aggregate_fit(self, server_round, results, failures):
        """聚合客户端模型"""
        if not results:
            return None
            
        # 获取客户端权重
        weights = []
        for client_id, fit_res in results:
            metrics = self.client_selector.client_metrics['performance'].get(client_id, {})
            weight = metrics.get('overall_score', 1.0)
            weights.append(weight)
            
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 加权聚合
        aggregated_weights = aggregate(results, weights)
        
        return aggregated_weights
        
    def evaluate(self, server_round, parameters):
        """评估全局模型"""
        loss, metrics = self.eval_fn(parameters)
        
        # 记录轮次指标
        self.round_metrics.append({
            'round': server_round,
            'loss': loss,
            'accuracy': metrics['accuracy'],
            'selected_clients': self.client_selector.current_round
        })
        
        # 打印评估结果
        print(f'轮次 {server_round}:')
        print(f'  损失: {loss:.4f}')
        print(f'  准确率: {metrics["accuracy"]:.4f}')
        print(f'  选中客户端数: {len(self.client_selector.current_round)}')
        
        return loss, metrics
        
    def get_training_metrics(self):
        """获取训练指标"""
        return {
            'round_metrics': self.round_metrics,
            'client_metrics': self.client_selector.client_metrics,
            'selection_history': list(self.client_selector.observation_window)
        } 