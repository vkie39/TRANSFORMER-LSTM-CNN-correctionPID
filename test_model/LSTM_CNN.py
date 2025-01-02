import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# CSV 데이터 읽기
data = pd.read_csv("motor_data.csv")
# 컬럼: ['target_speed', 'current_speed', 'voltage', 'current', 'motor_temp', 'ambient_temp', 'pid_value']
# 결측값 제거 (필요 시)
data = data.dropna()

# 정규화
scaler = MinMaxScaler()

# LSTM&CNN 입력에 사용할 특성. 입력 특성과 타겟 분리
features = ['target_speed', 'current_speed', 'voltage', 'current', 'motor_temp', 'ambient_temp']
target = ['pid_value']

# 정규화된 데이터 생성
scaled_features = scaler.fit_transform(data[features])
scaled_target = scaler.fit_transform(data[target])

# 정규화된 데이터를 DataFrame으로 변환 (편의를 위해)
scaled_data = pd.DataFrame(np.hstack([scaled_features, scaled_target]), columns=features + target)


# 데이터 전처리
def prepare_lstm_data(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length][features].values
        target = data.iloc[i + sequence_length][target].values
        x.append(seq)
        y.append(target)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

sequence_length = 10
# 데이터 준비
x_data, y_data = prepare_lstm_data(scaled_data, sequence_length)

# 학습/테스트 데이터 분리
train_size = int(len(x_data) * 0.8)
X_train, X_test = x_data[:train_size], x_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]


# LSTM-CNN 모델 정의
class LSTM_CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_CNN, self).__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1) # 1D CNN 레이어 추가
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝 출력 사용
        return out

# 모델 초기화
input_size = len(features)
hidden_size = 32
output_size = 1
model = LSTM_CNN(input_size, hidden_size, output_size)

# 손실 함수와 옵티마이저 정의
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 학습
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]

        # hidden state 초기화
        model.hidden = (torch.zeros(1, x_batch.size(0), hidden_size),
                        torch.zeros(1, x_batch.size(0), hidden_size))
                
        # 순전파
        outputs = model(x_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 실시간 제어 시뮬레이션
model.eval()
with torch.no_grad():
    recent_data = x_data[-1].unsqueeze(0)  # 마지막 시퀀스
    predicted_pid = model(recent_data).item()
    #정규화된 값을 다시 역변환
    predicted_pid_original = scaler.inverse_transform([[predicted_pid]])
    print(f"Predicted PID Value: {predicted_pid:.4f}")
    print(f"Predicted PID Value (Original Scale): {predicted_pid_original[0][0]:.4f}")

# 시각화
with torch.no_grad():
    predictions = model(x_data).squeeze().numpy()

plt.figure(figsize=(10, 6))
plt.plot(y_data.numpy(), label="True PID Values")
plt.plot(predictions, label="Predicted PID Values", linestyle="--")
plt.xlabel("Time Steps")
plt.ylabel("PID Value")
plt.legend()
plt.title("LSTM&CNN-based PID Prediction")
plt.grid()
plt.show()
