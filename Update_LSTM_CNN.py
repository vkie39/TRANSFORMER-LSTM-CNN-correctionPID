import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn

confirmed = pd.read_csv('우리가 받아들일 파일.csv')

def create_sequences(data, seq_length, input_columns, target_columns):
    xs = []
    ys = []
    
    for i in range(len(data) - seq_length):

        #입력이랑 타겟 분리
        x = data.iloc[i:(i+seq_length)][input_columns].values
        y = data.iloc[i + seq_length][target_columns].values

        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

input_columns = ['a', 'b', 'c', 'd'] #입력 시퀀스 이름
target_columns = ['e', 'f', 'g'] #출력

seq_length = 5 # 여기서 5는 총 시퀀스의 개수. 바꿔야 함

x, y = create_sequences(confirmed, seq_length, input_columns, target_columns)


#학습, 검증, 시험을 70, 20, 10
rowNum = confirmed.shape[0]
train_size = int(rowNum * 0.7)

x_train = x[:train_size]
y_train = y[:train_size]

val_size = int(rowNum * 0.2)
x_val = x[train_size:train_size+val_size]
y_val = y[train_size:train_size+val_size]

x_test = x[train_size+val_size:]
y_test = y[train_size+val_size:]


#데이터 스케일링
def MinMaxScale(array, min_val, max_val):
    return 2*(array - min_val) / (max_val - min_val) - 1

min = x_train.min()
max = x_train.max()

x_train = MinMaxScale(x_train, min, max)
y_train = MinMaxScale(y_train, min, max)
x_val = MinMaxScale(x_val, min, max)
y_val = MinMaxScale(y_val, min, max)
x_test = MinMaxScale(x_test, min, max)
y_test = MinMaxScale(y_test, min, max)

# Tensor로 변환
def make_Tensor(array):
    return torch.from_numpy(array).float()

x_train_final = make_Tensor(x_train)
y_train_final = make_Tensor(y_train)
x_val_final = make_Tensor(x_val)
y_val_final = make_Tensor(y_val)
x_test_final = make_Tensor(x_test)
y_test_final = make_Tensor(y_test)

#CNN-LSTM 모델 생성

class MotorPredict(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(MotorPredict, self).__init__() #초기화
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.c1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size = 2, stride = 1) # 1D CNN 레이어 추가
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    #새로운 시퀀스 처리하기 위해 이전 시퀀스(은닉 상태, 셀 상태) 상태 초기화
    def reset_hidden_state(self): 
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len-1, self.n_hidden)
        )
    
    def forward(self, sequences):
        sequences = self.c1(sequences.view(len(sequences), 1, -1))
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len-1, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len-1, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
    

#모델 학습
def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=100, verbose = 10, patience = 10):
    loss_fn = torch.nn.L1Loss() #
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    train_hist = []
    val_hist = []

    for t in range(num_epochs):
        epoch_loss = 0

        for idx, seq in enumerate(train_data): # sample 별 hidden state reset을 해줘야 함 
            
            model.reset_hidden_state()

            # train loss
            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx]) # 1개의 step에 대한 loss

            # update weights
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        train_hist.append(epoch_loss / len(train_data))

        if val_data is not None:

            with torch.no_grad():

                val_loss = 0

                for val_idx, val_seq in enumerate(val_data):

                    model.reset_hidden_state() #seq 별로 hidden state 초기화 

                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])

                    val_loss += val_step_loss
                
            val_hist.append(val_loss / len(val_data)) # val hist에 추가

            ## verbose 번째 마다 loss 출력 
            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

            ## patience 번째 마다 early stopping 여부 확인
            if (t % patience == 0) & (t != 0):
                
                ## loss가 커졌다면 early stop
                if val_hist[t - patience] < val_hist[t] :

                    print('\n Early Stopping')

                    break

        elif t % verbose == 0:
            print(f'Epoch {t} train loss: {epoch_loss / len(train_data)}')

            
    return model, train_hist, val_hist

model = MotorPredict(
    n_features=1,
    n_hidden=4,
    seq_len=seq_length,
    n_layers=1
)

print(model) #모델의 구조와 계층 정보 

#모델 학습
model, train_hist, val_hist = train_model(
    model,
    x_train_final,
    y_train_final,
    x_val_final,
    y_val_final,
    num_epochs=100,
    verbose=10,
    patience=50
)

# 테스트 데이터 평가 함수 추가 (이건 chat gpt _ 틀릴 수 있음)
def evaluate_model(model, test_data, test_labels):
    model.eval()  # 평가 모드
    loss_fn = torch.nn.L1Loss()  
    total_loss = 0
    
    with torch.no_grad():  # 그래디언트 계산 비활성화 (평가 시에는 필요 없음)
        for idx, seq in enumerate(test_data):
            model.reset_hidden_state()  # 시퀀스마다 hidden state 초기화
            seq = torch.unsqueeze(seq, 0)  # 배치 차원 추가
            y_pred = model(seq)  # 모델 예측
            loss = loss_fn(y_pred[0].float(), test_labels[idx])  # 손실 계산
            total_loss += loss.item()  # 손실 누적
    
    avg_loss = total_loss / len(test_data)  # 평균 손실 계산
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

#평가
test_loss = evaluate_model(model, x_test_final, y_test_final)

plt.plot(train_hist, label="Training loss")
plt.plot(val_hist, label="Val loss")
plt.legend()