import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_classes: int):
        """初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            num_classes: 类别数
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入数据，形状可以是：
               - [batch_size, input_size]
               - [batch_size, seq_len, input_size]
               - [batch_size, channels, seq_len, input_size]
            
        Returns:
            torch.Tensor: 输出预测
        """
        # 处理输入形状
        if len(x.shape) == 2:  # [batch_size, input_size]
            # 添加序列维度
            x = x.unsqueeze(1)
        elif len(x.shape) == 4:  # [batch_size, channels, seq_len, input_size]
            # 合并通道和序列维度
            batch_size, channels, seq_len, input_size = x.shape
            x = x.reshape(batch_size, channels * seq_len, input_size)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        
        return out
        
    def get_model_info(self):
        """获取模型信息
        
        Returns:
            Dict: 包含模型参数信息的字典
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_classes': self.num_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 