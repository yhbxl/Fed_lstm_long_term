import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import yaml

class MetricsAnalyzer:
    """指标分析器，用于分析实验结果和生成可视化"""
    
    def __init__(self, results_dir: str):
        """初始化指标分析器
        
        Args:
            results_dir: 结果目录路径
        """
        self.results_dir = results_dir
        self.metrics = self._load_metrics()
        self.config = self._load_config()
        
    def _load_metrics(self) -> Dict[str, Any]:
        """加载指标数据
        
        Returns:
            Dict[str, Any]: 指标数据
        """
        metrics = {}
        metrics_dir = os.path.join(self.results_dir, 'metrics')
        
        if not os.path.exists(metrics_dir):
            return metrics
            
        # 加载全局指标
        for filename in os.listdir(metrics_dir):
            if filename.endswith('.csv') and os.path.isfile(os.path.join(metrics_dir, filename)):
                metric_name = os.path.splitext(filename)[0]
                try:
                    df = pd.read_csv(os.path.join(metrics_dir, filename))
                    if not df.empty:
                        metrics[metric_name] = df[metric_name].values.tolist()
                except Exception as e:
                    print(f"加载指标 {metric_name} 时出错: {str(e)}")
        
        # 加载客户端指标
        client_metrics_dir = os.path.join(metrics_dir, 'clients')
        if os.path.exists(client_metrics_dir):
            metrics['client_metrics'] = {}
            
            for filename in os.listdir(client_metrics_dir):
                if filename.startswith('client_') and filename.endswith('.csv'):
                    client_id = int(filename.replace('client_', '').replace('.csv', ''))
                    try:
                        df = pd.read_csv(os.path.join(client_metrics_dir, filename))
                        metrics['client_metrics'][client_id] = df.to_dict('records')
                    except Exception as e:
                        print(f"加载客户端 {client_id} 指标时出错: {str(e)}")
        
        return metrics
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict[str, Any]: 配置数据
        """
        config_path = os.path.join(self.results_dir, 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"加载配置文件时出错: {str(e)}")
        
        return {}
    
    def plot_metric(self, metric_name: str, title: str = None, save_path: str = None):
        """绘制单个指标曲线
        
        Args:
            metric_name: 指标名称
            title: 图表标题
            save_path: 保存路径
        """
        if metric_name not in self.metrics:
            print(f"指标 {metric_name} 不存在")
            return
            
        values = self.metrics[metric_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-')
        
        plt.xlabel('轮次')
        plt.ylabel(metric_name)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'{metric_name} 随轮次变化')
            
        plt.grid(True)
        plt.tight_layout()
        
        # 设置x轴为整数
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.results_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'{metric_name}.png')
            
        plt.savefig(save_path)
        plt.close()
        print(f"图表已保存到: {save_path}")
    
    def plot_multiple_metrics(self, metric_names: List[str], title: str = None, save_path: str = None):
        """绘制多个指标曲线
        
        Args:
            metric_names: 指标名称列表
            title: 图表标题
            save_path: 保存路径
        """
        # 检查所有指标是否存在
        valid_metrics = [name for name in metric_names if name in self.metrics]
        if not valid_metrics:
            print("没有有效的指标可绘制")
            return
            
        plt.figure(figsize=(12, 7))
        
        for name in valid_metrics:
            values = self.metrics[name]
            plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-', label=name)
        
        plt.xlabel('轮次')
        plt.ylabel('指标值')
        
        if title:
            plt.title(title)
        else:
            plt.title('多指标对比')
            
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 设置x轴为整数
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.results_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, 'multiple_metrics.png')
            
        plt.savefig(save_path)
        plt.close()
        print(f"图表已保存到: {save_path}")
    
    def plot_client_selection_frequency(self, save_path: str = None):
        """绘制客户端选择频率热力图
        
        Args:
            save_path: 保存路径
        """
        if 'selected_clients' not in self.metrics:
            print("缺少客户端选择数据")
            return
            
        selected_clients = self.metrics['selected_clients']
        all_clients = set()
        
        # 获取所有客户端ID
        for round_selections in selected_clients:
            all_clients.update(round_selections)
            
        all_clients = sorted(list(all_clients))
        num_rounds = len(selected_clients)
        
        # 创建选择矩阵
        selection_matrix = np.zeros((len(all_clients), num_rounds))
        
        for round_idx, round_selections in enumerate(selected_clients):
            for client_id in round_selections:
                client_idx = all_clients.index(client_id)
                selection_matrix[client_idx, round_idx] = 1
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(selection_matrix, cmap='viridis', cbar_kws={'label': '是否选中'})
        
        plt.yticks(np.arange(len(all_clients)) + 0.5, all_clients)
        plt.xticks(np.arange(num_rounds) + 0.5, range(1, num_rounds + 1))
        plt.xlabel('轮次')
        plt.ylabel('客户端ID')
        plt.title('客户端选择频率')
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.results_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, 'client_selection_frequency.png')
            
        plt.savefig(save_path)
        plt.close()
        print(f"图表已保存到: {save_path}")
    
    def analyze_convergence(self, metric_name: str = 'val_accuracy', threshold: float = 0.9):
        """分析收敛性能
        
        Args:
            metric_name: 要分析的指标名称
            threshold: 收敛阈值
            
        Returns:
            Dict[str, Any]: 收敛分析结果
        """
        if metric_name not in self.metrics:
            print(f"指标 {metric_name} 不存在")
            return None
            
        values = self.metrics[metric_name]
        num_rounds = len(values)
        
        # 找到首次达到阈值的轮次
        convergence_round = None
        for i, value in enumerate(values):
            if value >= threshold:
                convergence_round = i + 1
                break
        
        # 计算稳定性（最后5轮的方差）
        stability = np.var(values[-5:]) if num_rounds >= 5 else None
        
        # 计算最终性能
        final_performance = values[-1] if values else None
        
        # 计算提升速度（每轮平均增长）
        improvement_rate = (values[-1] - values[0]) / num_rounds if num_rounds > 1 else 0
        
        result = {
            'convergence_round': convergence_round,
            'stability': stability,
            'final_performance': final_performance,
            'improvement_rate': improvement_rate,
            'num_rounds': num_rounds
        }
        
        return result
    
    def compare_strategies(self, results_dirs: Dict[str, str], metric_name: str, 
                          title: str = None, save_path: str = None):
        """比较不同策略的性能
        
        Args:
            results_dirs: 策略名称 -> 结果目录路径的字典
            metric_name: 要比较的指标名称
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 7))
        
        strategy_data = {}
        max_rounds = 0
        
        for strategy_name, result_dir in results_dirs.items():
            analyzer = MetricsAnalyzer(result_dir)
            if metric_name in analyzer.metrics:
                values = analyzer.metrics[metric_name]
                max_rounds = max(max_rounds, len(values))
                plt.plot(range(1, len(values) + 1), values, marker='o', linestyle='-', label=strategy_name)
                strategy_data[strategy_name] = values
        
        if not strategy_data:
            print(f"没有策略包含指标 {metric_name}")
            return
            
        plt.xlabel('轮次')
        plt.ylabel(metric_name)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'不同策略的{metric_name}对比')
            
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 设置x轴为整数
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlim(1, max_rounds)
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.results_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'strategy_comparison_{metric_name}.png')
            
        plt.savefig(save_path)
        plt.close()
        print(f"图表已保存到: {save_path}")
        
        # 返回比较数据
        return strategy_data
    
    def generate_report(self, output_path: str = None):
        """生成综合分析报告
        
        Args:
            output_path: 报告保存路径
        """
        if not self.metrics:
            print("没有可用的指标数据")
            return
            
        # 配置信息
        config_summary = {
            'data': self.config.get('data', {}),
            'model': self.config.get('model', {}),
            'client_selection': self.config.get('client_selection', {}),
            'training': self.config.get('training', {})
        }
        
        # 性能分析
        performance_metrics = {}
        
        if 'val_accuracy' in self.metrics:
            performance_metrics['accuracy'] = self.analyze_convergence('val_accuracy')
            
        if 'val_loss' in self.metrics:
            performance_metrics['loss'] = self.analyze_convergence('val_loss', threshold=0.1)
            
        if 'communication_cost' in self.metrics:
            performance_metrics['communication'] = {
                'total_cost': sum(self.metrics['communication_cost']),
                'avg_cost_per_round': sum(self.metrics['communication_cost']) / len(self.metrics['communication_cost'])
            }
        
        # 客户端选择分析
        client_selection_analysis = {}
        
        if 'selected_clients' in self.metrics:
            all_clients = set()
            for round_selections in self.metrics['selected_clients']:
                all_clients.update(round_selections)
                
            all_clients = sorted(list(all_clients))
            num_rounds = len(self.metrics['selected_clients'])
            
            # 计算每个客户端被选中的频率
            selection_frequency = {}
            for client_id in all_clients:
                count = sum(1 for round_selections in self.metrics['selected_clients'] if client_id in round_selections)
                selection_frequency[client_id] = count / num_rounds
                
            client_selection_analysis = {
                'num_unique_clients': len(all_clients),
                'avg_clients_per_round': sum(len(round_selections) for round_selections in self.metrics['selected_clients']) / num_rounds,
                'selection_frequency': selection_frequency
            }
        
        # 组装报告
        report = {
            'config': config_summary,
            'performance': performance_metrics,
            'client_selection': client_selection_analysis,
            'metrics_summary': {
                name: {
                    'min': min(values),
                    'max': max(values),
                    'avg': sum(values) / len(values),
                    'final': values[-1]
                } for name, values in self.metrics.items() if isinstance(values, list) and values
            }
        }
        
        # 保存报告
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'analysis_report.json')
            
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"分析报告已保存到: {output_path}")
        
        return report 