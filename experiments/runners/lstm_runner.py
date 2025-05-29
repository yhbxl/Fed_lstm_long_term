import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

from experiments.base.base_experiment import BaseExperiment
from core.federated_trainer import FederatedTrainer
from core.client_selector import LongTermClientSelector
from core.lstm_performance_predictor import LSTMPredictor
from core.models import get_model


class LSTMExperiment(BaseExperiment):
    """基于论文的联邦学习实验"""
    
    def __init__(self, args, output_dir: str = "results"):
        """初始化实验
        
        Args:
            args: 命令行参数对象
            output_dir: 输出目录路径
        """
        self.args = args
        self.metrics_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化数据和模型
        self.clients = None
        self.global_model = None
        self.test_loader = None
        
        # 训练参数
        self.num_rounds = args.num_rounds
        self.eval_interval = args.eval_interval
        
        # 初始化指标跟踪
        self.global_metrics = {
            'accuracy': [],
            'loss': []
        }
        self.client_metrics = {}
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"Experiment_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        self.save_config()
        
        # 设置组件
        self.setup_components()
        
    def create_model(self) -> nn.Module:
        """创建模型
        
        Returns:
            nn.Module: 创建的模型
        """
        model = get_model(self.args.model, self.args.num_classes)
        return model.to(self.device)
        
    def setup_components(self):
        """设置实验组件"""
        # 初始化模型
        self.model = self.create_model()
        
        # 初始化LSTM预测器
        self.lstm_predictor = LSTMPredictor(
            input_size=self.args.lstm_input_size,
            hidden_size=self.args.lstm_hidden_size,
            num_layers=self.args.lstm_num_layers,
            dropout=self.args.lstm_dropout
        )
        
        # 初始化客户端选择器
        self.client_selector = LongTermClientSelector(
            args=self.args
        )
        
        # 初始化训练器
        self.trainer = FederatedTrainer(
            args=self.args,
            eval_fn=self._create_eval_fn()
        )
        
        # 打印模型信息
        print("\n模型信息:")
        print(f"- 模型类型: {self.args.model}")
        print(f"- 输出类别数: {self.args.num_classes}")
        print(f"- 设备: {self.device}")
        
    def _create_eval_fn(self):
        """创建评估函数"""
        def eval_fn(model: nn.Module) -> Tuple[float, Dict[str, float]]:
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in self.test_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(self.test_loader)
            
            metrics = {'accuracy': accuracy, 'loss': avg_loss}
            self.metrics_history.append(metrics)
            
            return avg_loss, metrics
            
        return eval_fn
        
    def prepare_data(self):
        """准备实验数据"""
        from torchvision import datasets, transforms
        from torch.utils.data import random_split, DataLoader
        
        # 准备数据集
        if self.args.dataset.lower() == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('datasets/data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('datasets/data', train=False, transform=transform)
        elif self.args.dataset.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10('datasets/data', train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10('datasets/data', train=False, transform=transform)
        else:
            raise ValueError(f"不支持的数据集：{self.args.dataset}")
        
        # 设置全局测试集
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # 模拟非IID数据分布
        train_labels = np.array(train_dataset.targets if hasattr(train_dataset, 'targets') else train_dataset.train_labels)
        num_classes = len(np.unique(train_labels))
        
        # 使用Dirichlet分布模拟非IID数据
        client_data_indices = self._dirichlet_split(train_labels, self.args.num_clients, num_classes, self.args.alpha)
        
        # 创建客户端数据
        self.clients = []
        
        for i in range(self.args.num_clients):
            # 获取客户端的数据索引
            indices = client_data_indices[i]
            
            # 创建子数据集
            subset = torch.utils.data.Subset(train_dataset, indices)
            
            # 分割为训练集和验证集
            val_size = int(0.2 * len(subset))
            train_size = len(subset) - val_size
            client_train_dataset, client_val_dataset = random_split(
                subset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                client_train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True
            )
            
            val_loader = DataLoader(
                client_val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False
            )
            
            # 创建客户端
            client = {
                'id': i,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'device': self.device,
                'data_amount': len(client_train_dataset)
            }
            
            self.clients.append(client)
            
        print(f"数据准备完成，共有{self.args.num_clients}个客户端")
        
    def _dirichlet_split(self, labels, num_clients, num_classes, alpha):
        """使用Dirichlet分布划分数据"""
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        client_indices = [[] for _ in range(num_clients)]
        
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients), size=num_classes)
        
        for class_id, indices in enumerate(class_indices):
            np.random.shuffle(indices)
            
            client_props = proportions[class_id]
            client_props = np.array([max(p, 0.001) for p in client_props])
            client_props = client_props / client_props.sum()
            
            client_sample_sizes = (client_props * len(indices)).astype(int)
            client_sample_sizes[-1] = len(indices) - client_sample_sizes[:-1].sum()
            
            start_idx = 0
            for client_id, size in enumerate(client_sample_sizes):
                client_indices[client_id].extend(indices[start_idx:start_idx + size])
                start_idx += size
                
        return client_indices
    
    def save_config(self):
        """保存配置到文件"""
        config = vars(self.args)
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
    def run(self):
        """运行实验"""
        # 准备数据
        self.prepare_data()
        
        # 训练模型
        print("\n开始训练...")
        for round_idx in range(self.num_rounds):
            print(f"\n轮次 {round_idx + 1}/{self.num_rounds}")
            
            # 选择客户端
            selected_clients = self.client_selector.select_clients(
                available_clients=self.clients,
                num_selected=int(len(self.clients) * self.args.selection_fraction),
                lstm_predictor=self.lstm_predictor
            )
            
            # 训练选中的客户端
            for client in selected_clients:
                client_id = client['id']
                self.trainer.train_client(
                    client_id=client_id,
                    model=self.model,
                    train_loader=client['train_loader'],
                    val_loader=client['val_loader']
                )
                
                # 更新客户端历史
                performance = self.trainer.evaluate_client(
                    model=self.model,
                    val_loader=client['val_loader']
                )
                self.client_selector.update_client_history(
                    client_id=client_id,
                    metrics=performance
                )
                
                # 更新客户端指标
                if client_id not in self.client_metrics:
                    self.client_metrics[client_id] = {
                        'accuracy': [],
                        'loss': []
                    }
                self.client_metrics[client_id]['accuracy'].append(performance['accuracy'])
                self.client_metrics[client_id]['loss'].append(performance['loss'])
            
            # 评估全局模型
            if (round_idx + 1) % self.eval_interval == 0:
                loss, metrics = self.trainer.evaluate_global_model(self.model)
                print(f"全局模型性能 - 准确率: {metrics['accuracy']:.4f}, 损失: {loss:.4f}")
                
                # 更新全局指标
                self.global_metrics['accuracy'].append(metrics['accuracy'])
                self.global_metrics['loss'].append(metrics['loss'])
                
                # 调整观察窗口大小
                if round_idx > 0:
                    prev_accuracy = self.metrics_history[-2]['accuracy']
                    curr_accuracy = metrics['accuracy']
                    improvement = curr_accuracy - prev_accuracy
                    self.client_selector.adjust_window_size(improvement)
        
        # 分析结果
        self.analyze_results()
        
    def analyze_results(self):
        """分析实验结果"""
        # 绘制性能曲线
        self._plot_performance_curves()
        
        # 生成报告
        self.generate_report()
        
    def _plot_performance_curves(self):
        """绘制性能曲线"""
        # 提取指标
        rounds = range(1, len(self.metrics_history) + 1)
        accuracies = [m['accuracy'] for m in self.metrics_history]
        losses = [m['loss'] for m in self.metrics_history]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制准确率曲线
        ax1.plot(rounds, accuracies, 'b-', label='Accuracy')
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy over Rounds')
        ax1.grid(True)
        ax1.legend()
        
        # 绘制损失曲线
        ax2.plot(rounds, losses, 'r-', label='Loss')
        ax2.set_xlabel('Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss over Rounds')
        ax2.grid(True)
        ax2.legend()
        
        # 保存图表
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存图表
        save_path = os.path.join(self.output_dir, 'performance_curves.png')
        plt.savefig(save_path)
        plt.close()
        
    def generate_report(self):
        """生成实验报告"""
        # 设置字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制性能曲线
        plt.figure(figsize=(12, 6))
        
        # 绘制全局模型性能
        accuracies = [m['accuracy'] for m in self.metrics_history]
        losses = [m['loss'] for m in self.metrics_history]
        rounds = range(1, len(self.metrics_history) + 1)
        
        plt.plot(rounds, accuracies, label='Global Model Accuracy', marker='o')
        plt.plot(rounds, losses, label='Global Model Loss', marker='s')
        
        # 绘制客户端性能
        for client_id in self.client_metrics:
            plt.plot(rounds, self.client_metrics[client_id]['accuracy'], 
                    label=f'Client {client_id} Accuracy', 
                    linestyle='--', alpha=0.5)
        
        plt.xlabel('Rounds')
        plt.ylabel('Metrics')
        plt.title('Federated Learning Performance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存图片
        save_path = os.path.join(self.output_dir, 'performance_curves.png')
        plt.savefig(save_path)
        plt.close()
        
        # 生成报告
        report = {
            'total_rounds': self.args.num_rounds,
            'client_selection_fraction': self.args.selection_fraction,
            'final_global_accuracy': float(accuracies[-1]),
            'final_global_loss': float(losses[-1]),
            'client_performance': {
                str(client_id): {
                    'final_accuracy': float(metrics['accuracy'][-1]),
                    'final_loss': float(metrics['loss'][-1])
                }
                for client_id, metrics in self.client_metrics.items()
            }
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False) 