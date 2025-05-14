import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
from .feature_extractor import ModelFeatureExtractor

class ClientPerformancePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(ClientPerformancePredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # LSTM层
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 全连接层
        x = self.dropout(context)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return torch.sigmoid(x)

class ClientPerformanceAnalyzer:
    def __init__(self, feature_size, num_classes=10, window_size=10):
        self.feature_size = feature_size
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.predictor = ClientPerformancePredictor(feature_size)
        self.feature_extractor = ModelFeatureExtractor(num_classes)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        # 存储每个客户端的历史数据
        self.client_history = {}
        self.performance_history = {}
        
    def update_client_history(self, client_id, data_loader, model, device, performance_metrics):
        """更新客户端历史数据"""
        features = self.feature_extractor.extract_features(data_loader, model, device)
        
        if client_id not in self.client_history:
            self.client_history[client_id] = deque(maxlen=self.window_size)
            self.performance_history[client_id] = deque(maxlen=self.window_size)
            
        self.client_history[client_id].append(features)
        self.performance_history[client_id].append(performance_metrics)
        
    def calculate_stability_score(self, client_id):
        """计算客户端稳定性分数"""
        if client_id not in self.performance_history or len(self.performance_history[client_id]) < 2:
            return 0.0
            
        history = list(self.performance_history[client_id])
        
        # 计算性能变化
        performance_changes = []
        for i in range(1, len(history)):
            change = abs(history[i] - history[i-1])
            performance_changes.append(change)
            
        # 计算稳定性分数（变化越小，分数越高）
        stability_score = 1.0 / (1.0 + np.mean(performance_changes))
        
        # 考虑长期趋势
        if len(history) >= 3:
            trend = np.polyfit(range(len(history)), history, 1)[0]
            trend_factor = 1.0 / (1.0 + abs(trend))
            stability_score = 0.7 * stability_score + 0.3 * trend_factor
            
        return stability_score
        
    def predict_performance(self, client_id):
        """预测客户端未来性能"""
        if client_id not in self.client_history or len(self.client_history[client_id]) < self.window_size:
            return 0.0
            
        history = list(self.client_history[client_id])
        x = torch.stack(history).unsqueeze(0)
        
        self.predictor.eval()
        with torch.no_grad():
            performance_prob = self.predictor(x)
            
        return performance_prob.item()
        
    def train_predictor(self, client_id, actual_performance):
        """训练性能预测器"""
        if client_id not in self.client_history or len(self.client_history[client_id]) < self.window_size:
            return
            
        history = list(self.client_history[client_id])
        x = torch.stack(history).unsqueeze(0)
        y = torch.tensor([[float(actual_performance)]])
        
        self.predictor.train()
        self.optimizer.zero_grad()
        prediction = self.predictor(x)
        loss = self.criterion(prediction, y)
        loss.backward()
        self.optimizer.step()
        
    def get_client_metrics(self, client_id):
        """获取客户端综合指标"""
        if client_id not in self.client_history:
            return {
                'stability_score': 0.0,
                'performance_prediction': 0.0,
                'overall_score': 0.0,
                'feature_importance': None,
                'long_term_trend': 'unknown'
            }
            
        stability_score = self.calculate_stability_score(client_id)
        performance_pred = self.predict_performance(client_id)
        
        # 计算特征重要性
        if len(self.client_history[client_id]) > 0:
            features = torch.stack(list(self.client_history[client_id]))
            performance_metrics = torch.tensor(list(self.performance_history[client_id]))
            feature_importance = self.feature_extractor.calculate_feature_importance(
                features, performance_metrics
            )
        else:
            feature_importance = None
            
        # 分析长期趋势
        if len(self.performance_history[client_id]) >= 3:
            history = list(self.performance_history[client_id])
            trend = np.polyfit(range(len(history)), history, 1)[0]
            if trend > 0.01:
                long_term_trend = 'improving'
            elif trend < -0.01:
                long_term_trend = 'degrading'
            else:
                long_term_trend = 'stable'
        else:
            long_term_trend = 'unknown'
        
        # 计算综合得分（考虑稳定性和预测性能）
        overall_score = 0.6 * stability_score + 0.4 * performance_pred
        
        return {
            'stability_score': stability_score,
            'performance_prediction': performance_pred,
            'overall_score': overall_score,
            'feature_importance': feature_importance,
            'long_term_trend': long_term_trend
        }
        
    def analyze_training_stability(self, client_id):
        """分析训练稳定性"""
        if client_id not in self.performance_history or len(self.performance_history[client_id]) < 3:
            return {
                'is_stable': True,
                'trend': 'unknown',
                'volatility': 0.0,
                'feature_importance': None,
                'long_term_analysis': {
                    'trend': 'unknown',
                    'confidence': 0.0
                }
            }
            
        history = list(self.performance_history[client_id])
        
        # 计算趋势
        trend = np.polyfit(range(len(history)), history, 1)[0]
        
        # 计算波动性
        volatility = np.std(history)
        
        # 判断是否稳定
        is_stable = volatility < 0.1 and abs(trend) < 0.05
        
        # 计算特征重要性
        features = torch.stack(list(self.client_history[client_id]))
        performance_metrics = torch.tensor(history)
        feature_importance = self.feature_extractor.calculate_feature_importance(
            features, performance_metrics
        )
        
        # 长期分析
        if len(history) >= 5:
            # 使用多项式拟合进行长期趋势分析
            poly = np.polyfit(range(len(history)), history, 2)
            long_term_trend = 'improving' if poly[0] > 0 else 'degrading' if poly[0] < 0 else 'stable'
            # 计算拟合优度作为置信度
            y_pred = np.polyval(poly, range(len(history)))
            r2 = 1 - np.sum((history - y_pred) ** 2) / np.sum((history - np.mean(history)) ** 2)
            confidence = max(0, min(1, r2))
        else:
            long_term_trend = 'unknown'
            confidence = 0.0
        
        return {
            'is_stable': is_stable,
            'trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'volatility': volatility,
            'feature_importance': feature_importance,
            'long_term_analysis': {
                'trend': long_term_trend,
                'confidence': confidence
            }
        } 