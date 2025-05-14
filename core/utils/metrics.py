#计算准确率、损失等评估指标

import torch

def get_eval_fn(test_loader):
    def evaluate(parameters):
        model = Net()
        model.load_state_dict(parameters)
        model.eval()

        correct, total, loss_total = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for x, y in test_loader:
                out = model(x)
                loss = criterion(out, y)
                loss_total += loss.item()
                pred = torch.argmax(out, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        avg_loss = loss_total / len(test_loader)
        return avg_loss, accuracy
    return evaluate

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc2(self.relu(self.fc1(x)))
