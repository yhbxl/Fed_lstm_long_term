import os
import json
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class BaseExperiment(ABC):
    """所有实验的基类，提供通用功能"""
    
    def __init__(self, config_path: str, output_dir: str = "results"):
        """初始化基础实验类
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.config_path = config_path
        self.output_dir = self._setup_output_dir(output_dir)
        self.metrics_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载配置
        self.config = self._load_config()
        
        # 设置随机种子
        self._set_seed()
        
        # 初始化结果存储
        self.results = {
            'metrics': {},
            'model': None,
            'config': self.config,
            'client_metrics': {}
        }
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 如果配置路径是相对路径，则转换为绝对路径
        if not os.path.isabs(self.config_path):
            self.config_path = os.path.join(project_root, self.config_path)
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到配置文件：{self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"配置文件格式错误：{str(e)}")
                
    def _setup_output_dir(self, base_dir: str) -> str:
        """设置输出目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = self.__class__.__name__
        output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    def save_config(self):
        """保存实验配置"""
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True)
            
    def save_metrics(self, metrics: Dict[str, Any]):
        """保存实验指标"""
        self.metrics_history.append(metrics)
        metrics_path = os.path.join(self.output_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
            
    def log_metrics(self, metrics: Dict[str, Any]):
        """记录实验指标"""
        print(f"\n=== 轮次 {metrics.get('round', 'N/A')} ===")
        for key, value in metrics.items():
            if key != 'round':
                print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
                
    def _set_seed(self):
        """设置随机种子，确保实验可重复性"""
        if 'system' in self.config and 'seed' in self.config['system']:
            seed = self.config['system']['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def save_results(self):
        """保存实验结果"""
        # 保存指标数据为CSV
        metrics_dir = os.path.join(self.output_dir, 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        # 保存全局指标
        for metric_name, values in self.results['metrics'].items():
            if isinstance(values, list):
                df = pd.DataFrame({metric_name: values})
                df.to_csv(os.path.join(metrics_dir, f'{metric_name}.csv'), index=False)
        
        # 保存客户端指标
        if self.results['client_metrics']:
            client_metrics_dir = os.path.join(metrics_dir, 'clients')
            os.makedirs(client_metrics_dir, exist_ok=True)
            
            for client_id, metrics in self.results['client_metrics'].items():
                # 将客户端指标转换为DataFrame
                if isinstance(metrics, list) and metrics:
                    # 确保metrics是字典列表
                    if all(isinstance(m, dict) for m in metrics):
                        df = pd.DataFrame(metrics)
                        df.to_csv(os.path.join(client_metrics_dir, f'client_{client_id}.csv'), index=False)
        
        # 保存配置文件副本
        with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        # 保存模型
        if self.results['model'] is not None:
            model_path = os.path.join(self.output_dir, 'model.pth')
            torch.save(self.results['model'].state_dict(), model_path)
    
    def plot_metrics(self, metric_names: List[str], title: str = None, save_path: str = None):
        """绘制指标曲线
        
        Args:
            metric_names: 要绘制的指标名称列表
            title: 图表标题
            save_path: 保存路径，默认为output_dir/plots/metrics.png
        """
        if not metric_names or not all(name in self.results['metrics'] for name in metric_names):
            print("警告：部分指标不存在，无法绘图")
            return
        
        plt.figure(figsize=(10, 6))
        
        for name in metric_names:
            values = self.results['metrics'][name]
            plt.plot(range(1, len(values) + 1), values, label=name)
        
        plt.xlabel('轮次')
        plt.ylabel('指标值')
        
        if title:
            plt.title(title)
        else:
            plt.title('训练指标')
            
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, 'metrics.png')
        
        # 确保保存路径的目录存在
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
    
    def compare_strategies(self, strategy_results: Dict[str, Dict[str, List[float]]], 
                          metric_name: str, title: str = None, save_path: str = None):
        """比较不同策略的性能
        
        Args:
            strategy_results: 策略名称 -> {指标名称 -> 指标值列表}
            metric_name: 要比较的指标名称
            title: 图表标题
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 6))
        
        for strategy_name, results in strategy_results.items():
            if metric_name in results:
                values = results[metric_name]
                plt.plot(range(1, len(values) + 1), values, label=strategy_name)
        
        plt.xlabel('轮次')
        plt.ylabel(metric_name)
        
        if title:
            plt.title(title)
        else:
            plt.title(f'不同策略的{metric_name}对比')
            
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        if save_path is None:
            plots_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            save_path = os.path.join(plots_dir, f'comparison_{metric_name}.png')
            
        plt.savefig(save_path)
        plt.close()
    
    @abstractmethod
    def prepare_data(self):
        """准备实验数据，子类必须实现"""
        pass
    
    @abstractmethod
    def create_model(self):
        """创建模型，子类必须实现"""
        pass
    
    @abstractmethod
    def run(self):
        """运行实验，子类必须实现"""
        pass
        
    def analyze_results(self):
        """分析实验结果（需要子类实现）"""
        raise NotImplementedError("子类必须实现analyze_results方法")
        
    def generate_report(self):
        """生成实验报告（需要子类实现）"""
        raise NotImplementedError("子类必须实现generate_report方法")
        
    def cleanup(self):
        """清理实验资源"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()