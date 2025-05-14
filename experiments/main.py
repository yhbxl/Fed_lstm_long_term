import os
import argparse
from experiments.runners.lstm_runner import LSTMExperiment

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='联邦学习LSTM实验')
    parser.add_argument('--config', type=str, default='experiments/configs/lstm_config.yaml',
                        help='配置文件路径（默认：experiments/configs/lstm_config.yaml）')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录（默认：results）')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行实验
    print(f"使用配置文件 {args.config}")
    print(f"结果将保存到 {args.output_dir}")
    
    experiment = LSTMExperiment(config_path=args.config, output_dir=args.output_dir)
    experiment.run()
    
    print(f"实验完成，结果已保存到 {args.output_dir}")

if __name__ == "__main__":
    main()