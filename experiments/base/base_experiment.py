import os
import json
import yaml
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

class BaseExperiment:
    def __init__(self, config_path: str, output_dir: str = "results"):
        """初始化基础实验类
        
        Args:
            config_path: 配置文件路径
            output_dir: 输出目录
        """
        self.config = self._load_config(config_path)
        self.output_dir = self._setup_output_dir(output_dir)
        self.metrics_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 如果配置路径是相对路径，则转换为绝对路径
        if not os.path.isabs(config_path):
            config_path = os.path.join(project_root, config_path)
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到配置文件：{config_path}")
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
                
    def run(self):
        """运行实验（需要子类实现）"""
        raise NotImplementedError("子类必须实现run方法")
        
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