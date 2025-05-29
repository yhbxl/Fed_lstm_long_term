import os
import argparse
from experiments.runners.lstm_runner import LSTMExperiment

def args_parser():
    parser = argparse.ArgumentParser(description='联邦学习LSTM实验')
    
    # 实验基本配置
    parser.add_argument('--experiment_name', type=str, default='lstm_federated_learning',
                      help='实验名称')
    parser.add_argument('--output_dir', type=str, default='results',
                      help='输出目录')
    
    # 数据集配置
    parser.add_argument('--dataset', type=str, default='mnist',
                      help='数据集名称 (mnist/cifar10)')
    parser.add_argument('--num_classes', type=int, default=10,
                      help='类别数量')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    
    # 联邦学习配置
    parser.add_argument('--num_clients', type=int, default=10,
                      help='客户端总数')
    parser.add_argument('--num_rounds', type=int, default=10,
                      help='训练轮数')
    parser.add_argument('--selection_fraction', type=float, default=0.1,
                      help='每轮选择的客户端比例')
    parser.add_argument('--alpha', type=float, default=0.5,
                      help='Dirichlet分布参数')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='cnn',
                      help='模型类型 (cnn/mlp)')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='学习率')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='权重衰减')
    
    # LSTM预测器配置
    parser.add_argument('--lstm_input_size', type=int, default=1,
                      help='LSTM输入维度')
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                      help='LSTM隐藏层大小')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                      help='LSTM层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.2,
                      help='LSTM dropout率')
    parser.add_argument('--lstm_learning_rate', type=float, default=0.001,
                      help='LSTM学习率')
    parser.add_argument('--lstm_sequence_length', type=int, default=5,
                      help='LSTM序列长度')
    parser.add_argument('--lstm_patience', type=int, default=5,
                      help='LSTM早停耐心值')
    
    # 客户端选择器配置
    parser.add_argument('--window_size', type=int, default=5,
                      help='观察窗口大小')
    parser.add_argument('--min_window_size', type=int, default=3,
                      help='最小观察窗口大小')
    parser.add_argument('--max_window_size', type=int, default=10,
                      help='最大观察窗口大小')
    parser.add_argument('--window_adjust_threshold', type=float, default=0.01,
                      help='窗口调整阈值')
    parser.add_argument('--performance_weight', type=float, default=0.7,
                      help='性能权重')
    parser.add_argument('--delay_weight', type=float, default=0.3,
                      help='延迟权重')
    
    # 评估配置
    parser.add_argument('--eval_interval', type=int, default=5,
                      help='评估间隔')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                      help='随机种子')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = args_parser()
    
    # 设置随机种子
    import torch
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    experiment = LSTMExperiment(args, args.output_dir)
    experiment.run()

if __name__ == '__main__':
    main()