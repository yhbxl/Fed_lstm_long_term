import torch
import torch.nn as nn

class BandwidthPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=2, num_layers=1):
        super(BandwidthPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # 只使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)
        return prediction

class BandwidthPredictorManager:
    def __init__(self, num_clients, learning_rate=0.01):
        self.predictors = {}
        self.learning_rate = learning_rate
        self.history = {i: [] for i in range(num_clients)}
        
    def initialize_predictor(self, client_id):
        if client_id not in self.predictors:
            self.predictors[client_id] = {
                'model': BandwidthPredictor(),
                'optimizer': torch.optim.Adam(
                    self.predictors[client_id]['model'].parameters(),
                    lr=self.learning_rate
                )
            }
    
    def update_history(self, client_id, bandwidth):
        self.history[client_id].append(bandwidth)
        # 保持最近10个数据点
        if len(self.history[client_id]) > 10:
            self.history[client_id].pop(0)
    
    def predict_bandwidth(self, client_id):
        if client_id not in self.predictors:
            self.initialize_predictor(client_id)
            
        if len(self.history[client_id]) < 5:  # 需要至少5个历史数据点
            return None
            
        # 准备输入数据
        x = torch.tensor(self.history[client_id][-5:]).float().unsqueeze(0).unsqueeze(-1)
        
        # 预测
        self.predictors[client_id]['model'].eval()
        with torch.no_grad():
            prediction = self.predictors[client_id]['model'](x)
        return prediction.item()
    
    def train_predictor(self, client_id, actual_bandwidth):
        if client_id not in self.predictors:
            self.initialize_predictor(client_id)
            
        if len(self.history[client_id]) < 5:
            return
            
        # 准备训练数据
        x = torch.tensor(self.history[client_id][-5:]).float().unsqueeze(0).unsqueeze(-1)
        y = torch.tensor([actual_bandwidth]).float().unsqueeze(0)
        
        # 训练
        self.predictors[client_id]['model'].train()
        self.predictors[client_id]['optimizer'].zero_grad()
        prediction = self.predictors[client_id]['model'](x)
        loss = nn.MSELoss()(prediction, y)
        loss.backward()
        self.predictors[client_id]['optimizer'].step() 