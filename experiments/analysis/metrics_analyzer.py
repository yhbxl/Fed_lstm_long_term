import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from sklearn.metrics import mean_squared_error, r2_score

class MetricsAnalyzer:
    def __init__(self, metrics_file: str):
        """初始化指标分析器
        
        Args:
            metrics_file: 指标文件路径
        """
        self.metrics_file = metrics_file
        self.metrics_data = self._load_metrics()
        
    def _load_metrics(self) -> List[Dict[str, Any]]:
        """加载指标数据"""
        with open(self.metrics_file, 'r') as f:
            return json.load(f)
            
    def analyze_training_progress(self) -> Dict[str, Any]:
        """分析训练进度"""
        metrics_df = pd.DataFrame(self.metrics_data)
        
        analysis = {
            'accuracy': {
                'mean': metrics_df['accuracy'].mean(),
                'std': metrics_df['accuracy'].std(),
                'max': metrics_df['accuracy'].max(),
                'min': metrics_df['accuracy'].min(),
                'trend': self._calculate_trend(metrics_df['accuracy'])
            },
            'loss': {
                'mean': metrics_df['loss'].mean(),
                'std': metrics_df['loss'].std(),
                'min': metrics_df['loss'].min(),
                'max': metrics_df['loss'].max(),
                'trend': self._calculate_trend(metrics_df['loss'])
            }
        }
        
        return analysis
        
    def analyze_client_performance(self) -> Dict[str, Dict[str, Any]]:
        """分析客户端性能"""
        client_metrics = {}
        
        for round_data in self.metrics_data:
            for client_id, metrics in round_data.get('client_metrics', {}).items():
                if client_id not in client_metrics:
                    client_metrics[client_id] = {
                        'accuracy': [],
                        'loss': [],
                        'training_time': []
                    }
                    
                client_metrics[client_id]['accuracy'].append(metrics['accuracy'])
                client_metrics[client_id]['loss'].append(metrics['loss'])
                client_metrics[client_id]['training_time'].append(metrics['training_time'])
                
        analysis = {}
        for client_id, metrics in client_metrics.items():
            analysis[client_id] = {
                'accuracy': {
                    'mean': np.mean(metrics['accuracy']),
                    'std': np.std(metrics['accuracy']),
                    'trend': self._calculate_trend(metrics['accuracy'])
                },
                'loss': {
                    'mean': np.mean(metrics['loss']),
                    'std': np.std(metrics['loss']),
                    'trend': self._calculate_trend(metrics['loss'])
                },
                'training_time': {
                    'mean': np.mean(metrics['training_time']),
                    'std': np.std(metrics['training_time'])
                }
            }
            
        return analysis
        
    def analyze_selection_effectiveness(self) -> Dict[str, Any]:
        """分析选择策略效果"""
        selection_metrics = {
            'selection_time': [],
            'selected_clients': [],
            'performance_improvement': []
        }
        
        for i in range(1, len(self.metrics_data)):
            prev_round = self.metrics_data[i-1]
            curr_round = self.metrics_data[i]
            
            selection_metrics['selection_time'].append(
                curr_round.get('client_selection_time', 0)
            )
            selection_metrics['selected_clients'].append(
                len(curr_round.get('selected_clients', []))
            )
            selection_metrics['performance_improvement'].append(
                curr_round['accuracy'] - prev_round['accuracy']
            )
            
        analysis = {
            'selection_time': {
                'mean': np.mean(selection_metrics['selection_time']),
                'std': np.std(selection_metrics['selection_time'])
            },
            'selected_clients': {
                'mean': np.mean(selection_metrics['selected_clients']),
                'std': np.std(selection_metrics['selected_clients'])
            },
            'performance_improvement': {
                'mean': np.mean(selection_metrics['performance_improvement']),
                'std': np.std(selection_metrics['performance_improvement']),
                'positive_ratio': np.mean(np.array(selection_metrics['performance_improvement']) > 0)
            }
        }
        
        return analysis
        
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
            
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
            
    def plot_metrics(self, output_dir: str):
        """绘制指标图表"""
        metrics_df = pd.DataFrame(self.metrics_data)
        
        # 1. 准确率和损失曲线
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(metrics_df['round'], metrics_df['accuracy'], label='Accuracy')
        plt.title('Training Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(metrics_df['round'], metrics_df['loss'], label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
        plt.close()
        
        # 2. 客户端性能对比
        client_metrics = self.analyze_client_performance()
        plt.figure(figsize=(10, 6))
        
        client_ids = list(client_metrics.keys())
        accuracies = [metrics['accuracy']['mean'] for metrics in client_metrics.values()]
        stds = [metrics['accuracy']['std'] for metrics in client_metrics.values()]
        
        plt.bar(client_ids, accuracies, yerr=stds)
        plt.title('Client Performance Comparison')
        plt.xlabel('Client ID')
        plt.ylabel('Mean Accuracy')
        plt.xticks(client_ids)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'client_performance.png'))
        plt.close()
        
        # 3. 选择策略效果
        selection_analysis = self.analyze_selection_effectiveness()
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(selection_metrics['selection_time'], bins=20)
        plt.title('Selection Time Distribution')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        plt.hist(selection_metrics['selected_clients'], bins=len(set(selection_metrics['selected_clients'])))
        plt.title('Number of Selected Clients')
        plt.xlabel('Count')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 3)
        plt.hist(selection_metrics['performance_improvement'], bins=20)
        plt.title('Performance Improvement Distribution')
        plt.xlabel('Improvement')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'selection_analysis.png'))
        plt.close() 