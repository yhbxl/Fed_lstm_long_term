import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import yaml
from experiments.base.base_experiment import BaseExperiment
from core.fl_trainer.federated_trainer import FederatedTrainer
from core.client_selector.client_selector import ClientSelector
from core.models.lstm_model import LSTMModel
from core.data.data_loader import get_data_loaders

class LSTMExperiment(BaseExperiment):
    def __init__(self, config_path: str, output_dir: str):
        """初始化LSTM实验
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        super().__init__(config_path, output_dir)
        self.metrics_history = []  # 添加指标历史记录
        
        # 创建带时间戳的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"LSTMExperiment_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置文件
        self.save_config()
        
        # 设置组件
        self.setup_components()
        
    def setup_components(self):
        """设置实验组件"""
        # 初始化模型
        self.model = LSTMModel(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_classes=self.config['model']['num_classes']
        )
        
        # 初始化客户端选择器
        self.client_selector = ClientSelector(
            strategy=self.config['client_selection']['client_selection_strategy'],
            config=self.config
        )
        
        # 初始化训练器
        self.trainer = FederatedTrainer(
            client_selector=self.client_selector,
            eval_fn=self._create_eval_fn(),
            config=self.config
        )
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        print("\n模型信息:")
        for key, value in model_info.items():
            print(f"- {key}: {value}")
        
    def _create_eval_fn(self):
        """创建评估函数"""
        def eval_fn(model: nn.Module) -> Tuple[float, Dict[str, float]]:
            # 获取测试数据
            _, test_loader = get_data_loaders(
                dataset=self.config['data']['dataset'],
                batch_size=self.config['data']['batch_size'],
                num_clients=1,  # 评估时只需要一个客户端
                data_dir=self.config['data']['data_dir']  # 使用配置的数据目录
            )
            
            # 评估模型
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    if torch.cuda.is_available():
                        inputs = inputs.cuda()
                        targets = targets.cuda()
                    
                    outputs = model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    
                    total_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / len(test_loader)
            
            # 记录指标
            metrics = {'accuracy': accuracy, 'loss': avg_loss}
            self.metrics_history.append(metrics)
            
            return avg_loss, metrics
            
        return eval_fn
        
    def run(self):
        """运行实验"""
        print(f"\n开始实验：{self.__class__.__name__}")
        print(f"实验配置：dataset={self.config['data']['dataset']}, client_selection={self.config['client_selection']['client_selection_strategy']}")
        print(f"数据目录：{self.config['data']['data_dir']}")
        
        # 获取数据加载器
        train_loaders, _ = get_data_loaders(
            dataset=self.config['data']['dataset'],
            batch_size=self.config['data']['batch_size'],
            num_clients=self.config['data']['num_clients'],
            data_dir=self.config['data']['data_dir']  # 使用配置的数据目录
        )
        
        # 创建客户端列表
        clients = []
        for i, train_loader in enumerate(train_loaders):
            client = {
                'id': i,
                'train_loader': train_loader,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }
            clients.append(client)
        
        print(f"\n总客户端数: {len(clients)}")
        print(f"训练轮次: {self.config['training']['num_rounds']}")
        print(f"每轮选择客户端比例: {self.config['client_selection']['fraction_fit']}")
        
        # 训练模型
        start_time = datetime.now()
        best_model = self.trainer.train(clients, self.model)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # 保存最佳模型
        model_path = os.path.join(self.output_dir, "best_model.pth")
        torch.save(best_model.state_dict(), model_path)
        print(f"\n最佳模型已保存到: {model_path}")
        
        # 记录训练时间
        self.training_time = training_time
        print(f"总训练时间: {training_time:.2f} 秒")
        
        # 分析结果
        self.analyze_results()
        
        # 生成报告
        self.generate_report()
        
        return best_model
        
    def analyze_results(self):
        """分析实验结果"""
        print("\n分析实验结果...")
        
        # 1. 分析训练稳定性
        stability_analysis = {}
        for client_id in range(self.config['data']['num_clients']):
            if client_id in self.client_selector.client_history:
                history = list(self.client_selector.client_history[client_id])
                if len(history) >= 2:
                    accuracy_history = [h.get('accuracy', 0.0) for h in history]
                    loss_history = [h.get('loss', 1.0) for h in history]
                    
                    stability_analysis[str(client_id)] = {
                        'accuracy_stability': float(1.0 / (1.0 + np.std(accuracy_history) / np.mean(accuracy_history))),
                        'loss_stability': float(1.0 / (1.0 + np.std(loss_history) / np.mean(loss_history))),
                        'num_selections': len(history)
                    }
                    
        # 2. 分析客户端选择分布
        selection_counts = {}
        for client_id in range(self.config['data']['num_clients']):
            if client_id in self.client_selector.client_history:
                selection_counts[str(client_id)] = len(self.client_selector.client_history[client_id])
                
        # 3. 分析性能趋势
        if self.metrics_history:
            performance_trends = {
                'accuracy': {
                    'mean': float(np.mean([m.get('accuracy', 0.0) for m in self.metrics_history])),
                    'std': float(np.std([m.get('accuracy', 0.0) for m in self.metrics_history])),
                    'min': float(np.min([m.get('accuracy', 0.0) for m in self.metrics_history])),
                    'max': float(np.max([m.get('accuracy', 0.0) for m in self.metrics_history]))
                },
                'loss': {
                    'mean': float(np.mean([m.get('loss', 0.0) for m in self.metrics_history])),
                    'std': float(np.std([m.get('loss', 0.0) for m in self.metrics_history])),
                    'min': float(np.min([m.get('loss', 0.0) for m in self.metrics_history])),
                    'max': float(np.max([m.get('loss', 0.0) for m in self.metrics_history]))
                }
            }
        else:
            performance_trends = {
                'accuracy': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'loss': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            }
        
        # 4. 汇总分析结果
        analysis_results = {
            'stability_analysis': stability_analysis,
            'selection_distribution': selection_counts,
            'performance_trends': performance_trends,
            'total_rounds': len(self.metrics_history),
            'training_time': self.training_time,
            'dataset': self.config['data']['dataset'],
            'client_selection_strategy': self.config['client_selection']['client_selection_strategy'],
            'num_clients': self.config['data']['num_clients']
        }
        
        # 保存分析结果
        analysis_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        print(f"分析结果已保存到: {analysis_path}")
        
        # 绘制并保存性能曲线
        self._plot_performance_curves()
            
    def _plot_performance_curves(self):
        """绘制性能曲线"""
        if not self.metrics_history:
            return
        
        # 创建图表目录
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 提取指标
        rounds = range(1, len(self.metrics_history) + 1)
        accuracies = [m.get('accuracy', 0.0) for m in self.metrics_history]
        losses = [m.get('loss', 0.0) for m in self.metrics_history]
        
        # 1. 准确率曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, accuracies, marker='o')
        plt.title(f'准确率趋势 - {self.config["client_selection"]["client_selection_strategy"]}')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "accuracy_trend.png"))
        plt.close()
        
        # 2. 损失曲线
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, losses, marker='o', color='red')
        plt.title(f'损失趋势 - {self.config["client_selection"]["client_selection_strategy"]}')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "loss_trend.png"))
        plt.close()
        
        # 3. 客户端选择分布
        if hasattr(self.client_selector, 'client_history'):
            selection_counts = {}
            for client_id in range(self.config['data']['num_clients']):
                if client_id in self.client_selector.client_history:
                    selection_counts[client_id] = len(self.client_selector.client_history[client_id])
                else:
                    selection_counts[client_id] = 0
            
            client_ids = list(selection_counts.keys())
            counts = list(selection_counts.values())
            
            plt.figure(figsize=(12, 6))
            plt.bar(client_ids, counts)
            plt.title(f'客户端选择分布 - {self.config["client_selection"]["client_selection_strategy"]}')
            plt.xlabel('客户端 ID')
            plt.ylabel('被选择次数')
            plt.xticks(client_ids)
            plt.grid(True, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "client_selection_distribution.png"))
            plt.close()
            
        print(f"性能曲线已保存到: {plots_dir}")
        
    def generate_report(self):
        """生成实验报告"""
        print("\n生成实验报告...")
        
        # 1. 读取分析结果
        analysis_path = os.path.join(self.output_dir, "analysis_results.json")
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis_results = json.load(f)
            
        # 2. 生成报告
        report = []
        report.append("# LSTM联邦学习实验报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        report.append("\n## 1. 实验配置")
        report.append(f"- 数据集: {self.config['data']['dataset']}")
        report.append(f"- 客户端数量: {self.config['data']['num_clients']}")
        report.append(f"- 选择策略: {self.config['client_selection']['client_selection_strategy']}")
        report.append(f"- 训练轮次: {self.config['training']['num_rounds']}")
        report.append(f"- 每轮选择客户端比例: {self.config['client_selection']['fraction_fit']}")
        report.append(f"- 总训练时间: {analysis_results['training_time']:.2f} 秒")
        
        report.append("\n## 2. 模型信息")
        report.append(f"- 输入大小: {self.config['model']['input_size']}")
        report.append(f"- 隐藏层大小: {self.config['model']['hidden_size']}")
        report.append(f"- LSTM层数: {self.config['model']['num_layers']}")
        report.append(f"- 输出类别数: {self.config['model']['num_classes']}")
        
        report.append("\n## 3. 性能指标")
        acc_trends = analysis_results['performance_trends']['accuracy']
        loss_trends = analysis_results['performance_trends']['loss']
        report.append(f"- 平均准确率: {acc_trends['mean']:.4f} ± {acc_trends['std']:.4f}")
        report.append(f"- 最高准确率: {acc_trends['max']:.4f}")
        report.append(f"- 平均损失: {loss_trends['mean']:.4f} ± {loss_trends['std']:.4f}")
        
        report.append("\n## 4. 客户端选择分布")
        sorted_clients = sorted(analysis_results['selection_distribution'].items(), 
                               key=lambda x: int(x[0]))
        for client_id, count in sorted_clients:
            report.append(f"- 客户端 {client_id}: 被选择 {count} 次")
            
        report.append("\n## 5. 稳定性分析")
        for client_id, stability in analysis_results['stability_analysis'].items():
            report.append(f"\n### 客户端 {client_id}")
            report.append(f"- 准确率稳定性: {stability['accuracy_stability']:.4f}")
            report.append(f"- 损失稳定性: {stability['loss_stability']:.4f}")
            report.append(f"- 被选择次数: {stability['num_selections']}")
            
        report.append("\n## 6. 性能曲线")
        report.append("性能曲线已生成并保存在 plots 目录中：")
        report.append("- accuracy_trend.png: 准确率趋势")
        report.append("- loss_trend.png: 损失趋势")
        report.append("- client_selection_distribution.png: 客户端选择分布")
            
        # 3. 保存报告
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
            
        print(f"实验报告已保存到: {report_path}")
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题