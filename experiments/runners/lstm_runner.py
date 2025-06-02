import os
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

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
        print(f"- 数据集: {self.args.dataset.upper()}")
        print(f"- 数据分布: {'Non-IID' if self.args.alpha < 1.0 else 'IID'} (alpha={self.args.alpha})")
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
            
            return avg_loss, metrics
            
        return eval_fn
        
    def _evaluate_loader(self, model: nn.Module, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """评估模型在指定DataLoader上的性能"""
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
        
        return avg_loss, {'accuracy': accuracy, 'loss': avg_loss}
        
    def prepare_data(self):
        """准备实验数据"""
        from torchvision import datasets, transforms
        from torch.utils.data import random_split, DataLoader
        import os
        import numpy as np
        
        # 创建数据目录
        data_dir = 'datasets/data'
        os.makedirs(data_dir, exist_ok=True)
        
        # 准备数据集
        if self.args.dataset.lower() == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
        elif self.args.dataset.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
        elif self.args.dataset.lower() == 'fmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))  # FMNIST的均值和标准差
            ])
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_dir, train=False, transform=transform)
        else:
            raise ValueError(f"不支持的数据集：{self.args.dataset}")
        
        # 设置全局测试集
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # 设置全局训练集 (用于评估全局模型在整个训练集上的表现)
        self.global_train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)
        
        # 模拟非IID数据分布
        # 获取训练集标签，确保兼容不同数据集类型
        if hasattr(train_dataset, 'targets'):
            train_labels = np.array(train_dataset.targets)
        elif hasattr(train_dataset, 'train_labels'):
            train_labels = np.array(train_dataset.train_labels)
        elif isinstance(train_dataset.dataset, datasets.ImageFolder):
            train_labels = np.array(train_dataset.dataset.targets)[train_dataset.indices]
        else:
            # 尝试从DataLoader获取一个批次的标签来推断类别数，但这不准确
            # 更好的做法是为每个数据集明确定义如何获取标签和类别数
            print("警告: 未知数据集类型，无法准确获取标签和类别数进行非IID划分。")
            # 假设10个类别并继续，但这可能导致错误
            num_classes = 10
            # 尝试生成伪标签数组进行划分，这可能不代表真实分布
            print("警告: 使用伪标签数组进行非IID划分，结果可能不准确。")
            train_labels = np.random.randint(0, num_classes, size=len(train_dataset))
            
        # 确保 num_classes 被正确设置，如果之前推断失败
        if 'num_classes' not in locals() or num_classes is None:
            try:
                num_classes = len(np.unique(train_labels))
            except Exception as e:
                print(f"错误：无法确定数据集类别数量：{e}")
                raise ValueError("无法进行非IID划分，请检查数据集处理逻辑或提供类别数量。") from e

        print(f"数据集 {self.args.dataset.upper()}，类别数: {num_classes}")
        
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
            
            # 确保分割后的数据集不为空
            if train_size <= 0 or val_size <= 0:
                 print(f"警告: 客户端 {i} 数据量过少，无法创建有效的训练集和验证集。训练集大小: {train_size}, 验证集大小: {val_size}。跳过此客户端。")
                 continue # 跳过数据量过少的客户端

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
            
        print(f"数据准备完成，共有{len(self.clients)}个有效客户端")
        
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
        
        # 初始化选中客户端历史和指标
        self.selected_clients_history = []
        self.metrics_history = []
        self.client_metrics = {}
        
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
            
            # 记录选中的客户端
            selected_client_ids = [client['id'] for client in selected_clients]
            self.selected_clients_history.append(selected_client_ids)
        
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
                self.metrics_history.append(metrics)
                
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
        # 绘制性能曲线 (准确率和损失分开绘制)
        self._plot_performance_curves()
        
        # 生成报告
        self.generate_report()
        
    def _plot_performance_curves(self):
        """绘制性能曲线 (准确率和损失)"""
        # 提取全局指标
        accuracies = [m['accuracy'] for m in self.metrics_history]
        losses = [m['loss'] for m in self.metrics_history]
        
        # 使用实际的训练轮次作为横轴
        # 如果 eval_interval > 1, eval_rounds 会跳跃
        # 如果 eval_interval = 1, eval_rounds 会是 1, 2, 3...
        eval_rounds = [(i + 1) * self.args.eval_interval for i in range(len(self.metrics_history))]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制准确率曲线
        ax1.plot(eval_rounds, accuracies, 'b-', label='Accuracy')
        ax1.set_xlabel('Rounds')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy over Rounds')
        ax1.legend()
        ax1.grid(True)
        # 设置横轴范围和刻度
        ax1.set_xlim([0, self.args.num_rounds + 1])
        ax1.set_xticks(np.arange(0, self.args.num_rounds + 1, self.args.eval_interval))
        
        # 绘制损失曲线
        ax2.plot(eval_rounds, losses, 'r-', label='Loss')
        ax2.set_xlabel('Rounds')
        ax2.set_ylabel('Loss')
        ax2.set_title('Model Loss over Rounds')
        ax2.legend()
        ax2.grid(True)
        # 设置横轴范围和刻度
        ax2.set_xlim([0, self.args.num_rounds + 1])
        ax2.set_xticks(np.arange(0, self.args.num_rounds + 1, self.args.eval_interval))
        
        # 保存图表
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'performance_curves.png')
        plt.savefig(save_path)
        plt.close()
        
    def generate_report(self):
        """生成实验报告 (图表, CSV和JSON)"""
        # 设置字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取全局指标
        accuracies = [m['accuracy'] for m in self.metrics_history]
        losses = [m['loss'] for m in self.metrics_history]
        
        # 使用实际的训练轮次作为横轴
        # 如果 eval_interval > 1, eval_rounds 会跳跃
        # 如果 eval_interval = 1, eval_rounds 会是 1, 2, 3...
        eval_rounds = [(i + 1) * self.args.eval_interval for i in range(len(self.metrics_history))]
        
        # 如果最后一轮不是评估轮次，将最后一轮也添加到横轴，但不添加额外的指标点（保持图不连续）
        if len(eval_rounds) > 0 and eval_rounds[-1] < self.args.num_rounds:
             # 检查最后一轮是否已经被包含 (可能由于 eval_interval 整除 num_rounds)
             if eval_rounds[-1] != self.args.num_rounds:
                 eval_rounds.append(self.args.num_rounds)

        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. 绘制准确率和损失分开的图表
        if accuracies and losses:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 绘制准确率曲线
            ax1.plot(eval_rounds[:len(accuracies)], accuracies, 'b-', label='Accuracy')
            ax1.set_xlabel('Rounds')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy over Rounds')
            ax1.legend()
            ax1.grid(True)
            
            # 绘制损失曲线
            ax2.plot(eval_rounds[:len(losses)], losses, 'r-', label='Loss')
            ax2.set_xlabel('Rounds')
            ax2.set_ylabel('Loss')
            ax2.set_title('Model Loss over Rounds')
            ax2.legend()
            ax2.grid(True)
            
            # 保存图表
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, 'performance_curves.png')
            plt.savefig(save_path)
            plt.close()
            print(f"\n性能曲线图已保存到: {save_path}")
        else:
            print("\n没有足够的全局指标数据生成性能曲线图。")
        
        # 2. 生成报告 (JSON)
        report = {
            'total_rounds': self.args.num_rounds,
            'client_selection_fraction': self.args.selection_fraction,
            'final_global_accuracy': float(accuracies[-1]) if accuracies else None,
            'final_global_loss': float(losses[-1]) if losses else None,
            'client_performance': {
                str(client_id): {
                    'final_accuracy': float(metrics['accuracy'][-1]) if metrics.get('accuracy') and metrics['accuracy'] else None,
                    'final_loss': float(metrics['loss'][-1]) if metrics.get('loss') and metrics['loss'] else None
                }
                for client_id, metrics in self.client_metrics.items()
                if metrics.get('accuracy') or metrics.get('loss') # 包含有准确率或损失数据的客户端
            }
        }
        
        report_path = os.path.join(self.output_dir, 'experiment_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        print(f"\n实验报告 (JSON) 已保存到: {report_path}")

        # 3. 保存全局指标到CSV
        import pandas as pd
        
        global_metrics_data = {
            'round': eval_rounds[:len(self.metrics_history)],
            'test_accuracy': accuracies,
            'test_loss': losses
        }
        
        # 添加平均训练指标
        train_accuracies_list = []
        train_losses_list = []
        
        # 获取所有有效的客户端指标
        for client_id, metrics in self.client_metrics.items():
            if metrics.get('accuracy') and metrics.get('loss'):
                # 确保数据长度一致
                min_length = min(len(metrics['accuracy']), len(self.metrics_history))
                if min_length > 0:
                    train_accuracies_list.append(metrics['accuracy'][:min_length])
                    train_losses_list.append(metrics['loss'][:min_length])

        if train_accuracies_list and train_losses_list:
            # 确保所有列表长度一致
            min_length = min(len(acc_list) for acc_list in train_accuracies_list)
            train_accuracies_list = [acc_list[:min_length] for acc_list in train_accuracies_list]
            train_losses_list = [loss_list[:min_length] for loss_list in train_losses_list]
            
            # 计算每轮的平均训练准确率和损失
            avg_train_accuracy = [sum(acc_list) / len(acc_list) for acc_list in zip(*train_accuracies_list)]
            avg_train_loss = [sum(loss_list) / len(loss_list) for loss_list in zip(*train_losses_list)]
            
            # 添加到全局指标数据中
            global_metrics_data['train_accuracy'] = avg_train_accuracy
            global_metrics_data['train_loss'] = avg_train_loss

        # 添加选中的客户端信息（只包含评估轮次对应的选中客户端）
        if hasattr(self, 'selected_clients_history') and self.selected_clients_history:
             # 确保选中的客户端历史记录长度与总轮次一致
             if len(self.selected_clients_history) == self.args.num_rounds:
                  # 提取与评估轮次对应的选中客户端列表
                  eval_selected_clients = []
                  for i in range(len(self.metrics_history)):
                      round_index = (i + 1) * self.args.eval_interval - 1
                      if round_index < self.args.num_rounds:
                           eval_selected_clients.append(str(self.selected_clients_history[round_index]))
                      else:
                           eval_selected_clients.append('None') # 或者根据需要处理
                           
                  # 确保长度与全局指标评估次数一致
                  if len(eval_selected_clients) == len(self.metrics_history):
                       global_metrics_data['selected_clients'] = eval_selected_clients
                  else:
                       print("警告: selected_clients历史记录与全局指标评估次数不匹配，全局metrics.csv中将不包含选中客户端信息。")

        if global_metrics_data and 'round' in global_metrics_data and len(global_metrics_data['round']) > 0:
            # 确保所有要放入DataFrame的列长度一致
            max_len = len(global_metrics_data['round'])
            df_data_global = {}
            for key, value in global_metrics_data.items():
                 if isinstance(value, list) and len(value) == max_len:
                      df_data_global[key] = value
                 elif not isinstance(value, list):
                      print(f"警告: 全局指标 '{key}' 不是列表，或长度不一致，global_metrics.csv中将不包含此列。")

            if df_data_global and 'round' in df_data_global:
                 df_global = pd.DataFrame(df_data_global)
                 metrics_dir = os.path.join(self.output_dir, 'metrics')
                 os.makedirs(metrics_dir, exist_ok=True)
                 global_metrics_path = os.path.join(metrics_dir, 'global_metrics.csv')
                 df_global.to_csv(global_metrics_path, index=False)
                 print(f"\n全局指标已保存到: {global_metrics_path}")
            else:
                 print("\n没有足够有效数据生成global_metrics.csv文件。")

        else:
            print("\n没有全局指标数据，无法生成global_metrics.csv文件。")
            
        # 4. 保存客户端表现到CSV
        if self.client_metrics:
             client_performance_data = {'round': list(range(1, self.args.num_rounds + 1))} # 客户端数据以总轮次为准
             
             # 收集所有客户端ID
             all_client_ids = sorted(list(self.client_metrics.keys()))
             
             for client_id in all_client_ids:
                  client_acc = self.client_metrics[client_id].get('accuracy', [])
                  client_loss = self.client_metrics[client_id].get('loss', [])
                  
                  # 用NaN填充缺失的数据点，使其长度与总轮次一致
                  acc_padded = client_acc + [np.nan] * (self.args.num_rounds - len(client_acc))
                  loss_padded = client_loss + [np.nan] * (self.args.num_rounds - len(client_loss))
                  
                  client_performance_data[f'client_{client_id}_accuracy'] = acc_padded
                  client_performance_data[f'client_{client_id}_loss'] = loss_padded

             # 添加选中的客户端信息到客户端性能文件
             if hasattr(self, 'selected_clients_history') and self.selected_clients_history:
                 if len(self.selected_clients_history) == self.args.num_rounds:
                      client_performance_data['selected_clients'] = [str(clients) for clients in self.selected_clients_history]
                 else:
                      print("警告: selected_clients历史记录与总轮次不匹配，客户端metrics.csv中将不包含选中客户端信息。")

             if client_performance_data and len(client_performance_data['round']) == self.args.num_rounds:
                  df_clients = pd.DataFrame(client_performance_data)
                  metrics_dir = os.path.join(self.output_dir, 'metrics')
                  os.makedirs(metrics_dir, exist_ok=True)
                  client_metrics_path = os.path.join(metrics_dir, 'client_performance.csv')
                  df_clients.to_csv(client_metrics_path, index=False)
                  print(f"\n客户端表现已保存到: {client_metrics_path}")
             else:
                  print("\n没有足够有效数据生成client_performance.csv文件。")
        else:
             print("\n没有客户端指标数据，无法生成client_performance.csv文件。") 