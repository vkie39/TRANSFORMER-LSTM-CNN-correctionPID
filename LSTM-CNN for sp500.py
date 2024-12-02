import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch import nn

confirmed = pd.read_csv('sample_data_sp500.csv')
#print(confirmed.columns) 
confirmed = confirmed.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

#confirmed = confirmed.select_dtypes(include=[np.number]) #이제 열 이름 빼는 거

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

input_columns = ['Open','High','Low'] #입력 시퀀스 이름
target_columns = ['Close'] #출력

seq_length = 5 # 여기서 5는 총 시퀀스의 개수. 바꿔야 함

x, y = create_sequences(confirmed, seq_length, input_columns, target_columns)

#train, test data
train_size = int(len(x) * 0.8)  # 80%는 학습, 20%는 테스트
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 결과 확인
print(f"Train set: {x_train.shape}, Test set: {x_test.shape}")


#데이터 스케일링
#x" =(1−(−1))⋅x ′ −1=2⋅x ′  −1
def MinMaxScale(array, min_val, max_val):
    return 2*(array - min_val) / (max_val - min_val) - 1

min = x_train.min()
max = x_train.max()

#validation loss랑 training loss가 너무 높음. 
print("x_train min:", x_train.min(), "max:", x_train.max())
print("y_train min:", y_train.min(), "max:", y_train.max())

x_train = MinMaxScale(x_train, min, max)
y_train = MinMaxScale(y_train, min, max)
x_test = MinMaxScale(x_test, min, max)
y_test = MinMaxScale(y_test, min, max)

# Tensor로 변환
def make_Tensor(array):
    return torch.from_numpy(array).float()

x_train_final = make_Tensor(x_train)
y_train_final = make_Tensor(y_train)
x_test_final = make_Tensor(x_test)
y_test_final = make_Tensor(y_test)


#CNN-LSTM 모델 생성
class MotorPredict(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(MotorPredict, self).__init__()
        self.n_hidden = 32
        self.seq_len = seq_len
        self.n_layers = n_layers
        self.c1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=2, stride=1)  # Conv1d
        self.dropout = nn.Dropout(p=0.5)
        self.linear_before_lstm = nn.Linear(14, 32)  # Conv1d 출력 크기를 LSTM 입력 크기로 변환
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=32,
            num_layers=n_layers,
            batch_first=True
        )
        self.linear = nn.Linear(in_features=32, out_features=1)

    '''
    #새로운 시퀀스 처리하기 위해 이전 시퀀스(은닉 상태, 셀 상태) 상태 초기화
    def reset_hidden_state(self): 
        self.hidden = (
            torch.zeros(self.n_layers, 1, self.n_hidden),
            torch.zeros(self.n_layers, 1, self.n_hidden)
        )
        '''
    '''
    #RuntimeError: shape '[1, 4, -1]' is invalid for input of size 14 오류가 남
    #self.seq_len-1를 Conv1d 출력 크기 계산해서 하도록 변경
    def forward(self, sequences):   
        sequences = self.c1(sequences.view(len(sequences), 1, -1)) #Conv1D에 입력
        #print("Conv1d의 sequences 크기: ", sequences.shape)

        #conv1d 출력 크기
        conv_output_length = sequences.shape[2] # << 마지막 차원 길이 
        #conv_output_length = (self.seq_len - self.c1.kernel_size[0]

        
        # Conv1d 출력 크기 계산
        #sequences = sequences.permute(0, 2, 1)
        #sequences = self.linear_before_lstm(sequences)
        

        sequences = sequences.view(len(sequences), conv_output_length, -1)
        lstm_out, self.hidden = self.lstm(sequences, self.hidden)


        #LSTM입력으로 시퀀스 변환하는 거 추가함
        sequences = sequences.view(len(sequences), conv_output_length, -1)

        lstm_out, self.hidden = self.lstm(sequences, self.hidden)

        #lstm의 마지막 타임스텝 출력 사용
        last_time_step = lstm_out.view(conv_output_length, len(sequences), self.n_hidden)[-1] 
        y_pred = self.linear(last_time_step)
        return y_pred
        '''
    
    def forward(self, sequences):
        sequences = self.c1(sequences.view(len(sequences), 1, -1))
        sequences = sequences.permute(0, 2, 1)
        lstm_out, _ = self.lstm(sequences)
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred


#모델 학습
def train_model(model, train_data, train_labels, val_data=None, val_labels=None, num_epochs=10, verbose = 10, validation_split=0.1):
    loss_fn = torch.nn.L1Loss() #
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

model = MotorPredict(
    n_features=1,
    n_hidden=32,
    seq_len=seq_length,
    n_layers=1
)

print(model) #모델의 구조와 계층 정보 

#모델 학습
model, train_hist, val_hist = train_model(
    model,
    x_train_final,
    y_train_final,
    num_epochs=200,
    verbose=10,
    validation_split=0.1
)

# 테스트 데이터 평가 함수 추가 (이건 chat gpt _ 틀릴 수 있음)
def evaluate_model(model, test_data, test_labels):
    model.eval()  # 평가 모드
    loss_fn = torch.nn.L1Loss()  
    total_loss = 0

    with torch.no_grad(): # 그래디언트 계산 비활성화 (평가 시에는 필요 없음)
        for idx in range(len(test_data)):
            y_pred = model(test_data[idx].unsqueeze(0)) #모델 예측
            loss = loss_fn(y_pred, test_labels[idx].unsqueeze(0)) #손실 계산
            total_loss += loss.item() #손실 누적
    
    avg_loss = total_loss / len(test_data)  # 평균 손실 계산
    print(f"Test Loss: {avg_loss:.4f}")
    return avg_loss

#평가
test_loss = evaluate_model(model, x_test_final, y_test_final)

# 모델 평가 결과 출력 함수
def print_model_performance(model, x_test, y_test):
    print("Evaluating model on test data...")
    test_loss = evaluate_model(model, x_test, y_test)  # 평균 손실
    
    # 추가 성능 지표 계산
    y_pred_list = []
    y_actual_list = []
    
    with torch.no_grad():
        for idx in range(len(x_test)):
            y_pred = model(x_test[idx].unsqueeze(0))
            y_pred_list.append(y_pred.squeeze().item())
            y_actual_list.append(y_test[idx].squeeze().item())
    
    y_pred_array = np.array(y_pred_list)
    y_actual_array = np.array(y_actual_list)
    
    mae = np.mean(np.abs(y_pred_array - y_actual_array))  # Mean Absolute Error
    mse = np.mean((y_pred_array - y_actual_array) ** 2)  # Mean Squared Error
    correlation = np.corrcoef(y_pred_array, y_actual_array)[0, 1]  # Correlation coefficient
    
    print(f"Test Loss (L1): {test_loss:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}")

# 모델 평가 및 성능 출력
print_model_performance(model, x_test_final, y_test_final)

plt.plot(train_hist, label="Training loss")
plt.plot(val_hist, label="Val loss")
plt.legend()
plt.show()
