#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # 系统参数
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 数据参数
    parser.add_argument('--dataset', type=str, default='cifar10', help='数据集名称: mnist, cifar10, emnist')
    parser.add_argument('--num_classes', type=int, default=10, help='类别数量')
    parser.add_argument('--num_clients', type=int, default=100, help='客户端数量')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--alpha', type=float, default=0.5, help='Dirichlet分布参数，控制非IID程度')
    
    # 模型参数
    parser.add_argument('--model', type=str, default='cnn', help='模型类型: cnn 或 mlp')
    parser.add_argument('--dim_in', type=int, default=784, help='输入维度')
    parser.add_argument('--dim_hidden', type=int, default=200, help='隐藏层维度')
    parser.add_argument('--dim_out', type=int, default=10, help='输出维度')
    parser.add_argument('--num_channels', type=int, default=1, help='输入通道数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    
    # LSTM预测器参数
    parser.add_argument('--lstm_input_size', type=int, default=1, help='LSTM输入维度')
    parser.add_argument('--lstm_hidden_size', type=int, default=128, help='LSTM隐藏层维度')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--lstm_dropout', type=float, default=0.2, help='LSTM Dropout率')
    parser.add_argument('--lstm_sequence_length', type=int, default=5, help='历史序列长度')
    parser.add_argument('--lstm_learning_rate', type=float, default=0.001, help='LSTM学习率')
    parser.add_argument('--lstm_patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--lstm_min_delta', type=float, default=0.001, help='早停最小改善阈值')
    parser.add_argument('--lstm_window_size', type=int, default=5, help='观察窗口大小')
    
    # 客户端选择参数
    parser.add_argument('--initial_window_size', type=int, default=5, help='初始观察窗口大小')
    parser.add_argument('--min_window_size', type=int, default=3, help='最小窗口大小')
    parser.add_argument('--max_window_size', type=int, default=10, help='最大窗口大小')
    parser.add_argument('--window_adjustment_threshold', type=float, default=0.05, help='窗口调整阈值')
    parser.add_argument('--history_weight', type=float, default=0.6, help='历史性能权重')
    parser.add_argument('--prediction_weight', type=float, default=0.4, help='预测性能权重')
    parser.add_argument('--adaptation_rate', type=float, default=0.1, help='权重调整率')
    parser.add_argument('--delay_penalty_factor', type=float, default=0.5, help='延迟惩罚因子')
    parser.add_argument('--max_delay_threshold', type=float, default=1.0, help='最大延迟阈值')
    parser.add_argument('--min_clients', type=int, default=5, help='每轮最少选择的客户端数量')
    parser.add_argument('--max_clients', type=int, default=20, help='每轮最多选择的客户端数量')
    parser.add_argument('--selection_fraction', type=float, default=0.2, help='每轮选择的客户端比例')
    
    # 训练参数
    parser.add_argument('--num_rounds', type=int, default=100, help='联邦学习总轮数')
    parser.add_argument('--local_epochs', type=int, default=5, help='本地训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学习率')
    parser.add_argument('--lr_decay', type=float, default=0.995, help='学习率衰减率')
    parser.add_argument('--min_lr', type=float, default=0.0001, help='最小学习率')
    parser.add_argument('--momentum', type=float, default=0.9, help='动量')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--eval_interval', type=int, default=1, help='评估间隔')
    
    # 通信参数
    parser.add_argument('--compression_ratio', type=float, default=0.1, help='梯度压缩比例')
    parser.add_argument('--gradient_threshold', type=float, default=0.01, help='梯度阈值')
    parser.add_argument('--bandwidth_min', type=int, default=10, help='最小带宽(kbps)')
    parser.add_argument('--bandwidth_max', type=int, default=10000, help='最大带宽(kbps)')
    parser.add_argument('--high_latency_ratio', type=float, default=0.3, help='高延迟设备比例')
    
    # 带宽预测器参数
    parser.add_argument('--bw_initial_window_size', type=int, default=5, help='带宽预测初始窗口大小')
    parser.add_argument('--bw_min_window_size', type=int, default=3, help='带宽预测最小窗口大小')
    parser.add_argument('--bw_max_window_size', type=int, default=10, help='带宽预测最大窗口大小')
    parser.add_argument('--bw_learning_rate', type=float, default=0.001, help='带宽预测学习率')
    parser.add_argument('--bw_adaptation_rate', type=float, default=0.1, help='带宽预测窗口调整率')
    parser.add_argument('--bw_latency_threshold', type=float, default=0.5, help='带宽预测延迟阈值')
    parser.add_argument('--bw_max_bandwidth', type=float, default=10.0, help='最大带宽')
    parser.add_argument('--bw_max_latency', type=float, default=1.0, help='最大延迟')
    
    # 添加训练相关参数
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--performance_threshold', type=float, default=0.8)
    
    args = parser.parse_args()
    return args 