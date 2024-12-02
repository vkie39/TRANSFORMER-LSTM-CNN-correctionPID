import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time  # 시간 측정을 위한 라이브러리

from torch import nn

confirmed = pd.read_csv('sample_data_sp500.csv')
confirmed = confirmed.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

def create_sequences(data, seq_length, input_columns, target_columns):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)][input_columns].values
        y = data.iloc[i + seq_length][target_columns].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

input_columns = ['Open', 'High', 'Low']
target_columns = ['Close']
seq_length = 5

x, y = create_sequences(confirmed, seq_length, input_columns, target_columns)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

def MinMaxScale(array, min_val, max_val):
    return 2 * (array - min_val) / (max_val - min_val) - 1

min_val = x_train.min()
max_val = x_train.max()

x_train = MinMaxScale(x_train, min_val, max_val)
y_train = MinMaxScale(y_train, min_val, max_val)
x_test = MinMaxScale(x_test, min_val, max_val)
y_test = MinMaxScale(y_test, min_val, max_val)

def make_Tensor(array):
    return torch.from_numpy(array).float()

x_train_final = make_Tensor(x_train)
y_train_final = make_Tensor(y_train)
x_test_final = make_Tensor(x_test)
y_test_final = make_Tensor(y_test)

class MotorPredict(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(MotorPredict, self).__init__()
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1)
        self.lstm = nn.LSTM(input_size=32, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def forward(self, sequences):
        sequences = self.c1(sequences.view(len(sequences), 1, -1))
        sequences = sequences.permute(0, 2, 1)
        lstm_out, _ = self.lstm(sequences)
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=10, verbose=10, validation_split=0.1):
    loss_fn = torch.nn.L1Loss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_size = int(len(train_data) * (1 - validation_split))
    val_data = train_data[train_size:]
    val_labels = train_labels[train_size:]
    train_data = train_data[:train_size]
    train_labels = train_labels[:train_size]

    train_hist = []
    val_hist = []

    for i in range(num_epochs):
        model.train()
        epoch_loss = 0

        for idx in range(len(train_data)):
            optimiser.zero_grad()
            y_pred = model(train_data[idx].unsqueeze(0))
            loss = loss_fn(y_pred, train_labels[idx].unsqueeze(0))
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()

        train_hist.append(epoch_loss / len(train_data))
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for idx in range(len(val_data)):
                y_pred = model(val_data[idx].unsqueeze(0))
                loss = loss_fn(y_pred, val_labels[idx].unsqueeze(0))
                val_loss += loss.item()

        val_hist.append(val_loss / len(val_data))

        if i % verbose == 0:
            print(f"Epoch {i}, Train Loss: {train_hist[-1]:.4f}, Validation Loss: {val_hist[-1]:.4f}")

    return model, train_hist, val_hist

model = MotorPredict(n_features=1, n_hidden=32, seq_len=seq_length, n_layers=1)

model, train_hist, val_hist = train_model(model, x_train_final, y_train_final, num_epochs=200, verbose=10, validation_split=0.1)

def evaluate_model(model, test_data, test_labels):
    model.eval()
    loss_fn = torch.nn.L1Loss()
    total_loss = 0

    with torch.no_grad():
        for idx in range(len(test_data)):
            y_pred = model(test_data[idx].unsqueeze(0))
            loss = loss_fn(y_pred, test_labels[idx].unsqueeze(0))
            total_loss += loss.item()

    avg_loss = total_loss / len(test_data)
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

test_loss = evaluate_model(model, x_test_final, y_test_final)

# 개별 샘플 예측 시간 측정
single_prediction_times = []

for i in range(len(x_test_final)):
    single_input = x_test_final[i].unsqueeze(0)
    start_time = time.time()  # 시작 시간
    model(single_input)  # 예측
    end_time = time.time()  # 종료 시간
    single_prediction_times.append(end_time - start_time)  # 소요 시간 기록

average_prediction_time = np.mean(single_prediction_times)
print(f"Average Prediction Time per Sample: {average_prediction_time:.6f} seconds")

# 학습 및 검증 손실 그래프
plt.plot(train_hist, label="Training loss")
plt.plot(val_hist, label="Validation loss")
plt.legend()
plt.show()
