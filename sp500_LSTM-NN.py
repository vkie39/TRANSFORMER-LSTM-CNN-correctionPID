import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import psutil  # RAM 및 CPU 사용량 확인용
import os
import time

# RAM 사용량 확인 함수
def print_ram_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 ** 2)  # Resident Set Size (MB)
    print(f"RAM Usage: {rss_memory:.2f} MB")

# CPU 사용량 확인 함수
def print_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)  # 1초 간격으로 CPU 사용량 측정
    print(f"CPU Usage: {cpu_percent:.2f}%")

# CSV 파일 로드
data = pd.read_csv("sample_data_sp500.csv")
data = data.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

# 입력과 출력 변수 분리
features = ['Open', 'High', 'Low']
target = 'Close'

X = data[features].values
y = data[target].values

# 입력 데이터와 타겟 데이터 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)  # 입력 데이터 정규화
y = target_scaler.fit_transform(y.reshape(-1, 1))  # 타겟 데이터 정규화

seq_length = 5  # 시퀀스 길이

# Sliding Window를 이용한 시퀀스 데이터 생성
def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_sliding_window(X, y, seq_length)

# 슬라이딩 윈도우 결과 확인
print(f"Generated X_seq shape: {X_seq.shape}")
print(f"Generated y_seq shape: {y_seq.shape}")

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# LSTM + Fully Connected Neural Network (NN) 모델 정의
class LSTMNNModel(tf.keras.Model):
    def __init__(self, lstm_units, dense_units, dropout_rate=0.1):
        super(LSTMNNModel, self).__init__()
        self.lstm = layers.LSTM(units=lstm_units, return_sequences=False)  # LSTM 레이어
        self.dropout = layers.Dropout(dropout_rate)  # Dropout 레이어
        self.dense1 = layers.Dense(units=dense_units, activation="relu")  # Fully Connected 레이어
        self.dense2 = layers.Dense(1, activation="linear")  # 출력 레이어

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)  # LSTM 처리
        x = self.dropout(x, training=training)   # Dropout 적용
        x = self.dense1(x)                       # 첫 번째 Dense 레이어
        return self.dense2(x)                    # 출력 레이어

# 모델 초기화
model = LSTMNNModel(lstm_units=64, dense_units=32, dropout_rate=0.2)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",  # Mean Squared Error
              metrics=["mae"])

# RAM 및 CPU 사용량 확인 (학습 시작 전)
print("Before Training:")
print_ram_usage()
print_cpu_usage()

# 모델 학습
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=32,
                    shuffle=True)

# RAM 및 CPU 사용량 확인 (학습 후)
print("After Training:")
print_ram_usage()
print_cpu_usage()

# 모델 평가
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# RAM 및 CPU 사용량 확인 (평가 후)
print("After Evaluation:")
print_ram_usage()
print_cpu_usage()

# 개별 샘플 예측 시간 및 리소스 사용량 측정
single_prediction_times = []
prediction_ram_usage = []
prediction_cpu_usage = []

for i in range(len(X_test)):
    single_input = X_test[i:i+1]  # 개별 입력 데이터
    start_time = time.time()
    
    # 예측 수행
    model.predict(single_input)
    
    end_time = time.time()
    single_prediction_times.append(end_time - start_time)
    
    # RAM 및 CPU 사용량 기록
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 ** 2)  # MB 단위
    cpu_percent = psutil.cpu_percent(interval=0.1)  # 짧은 간격으로 CPU 사용량 측정
    prediction_ram_usage.append(rss_memory)
    prediction_cpu_usage.append(cpu_percent)

# 평균 예측 시간 계산
average_prediction_time = np.mean(single_prediction_times)
print(f"Average Prediction Time per Sample: {average_prediction_time:.6f} seconds")

# 예측 중 리소스 사용량 출력
print(f"Prediction RAM Usage (Max): {max(prediction_ram_usage):.2f} MB")
print(f"Prediction RAM Usage (Avg): {np.mean(prediction_ram_usage):.2f} MB")
print(f"Prediction CPU Usage (Max): {max(prediction_cpu_usage):.2f}%")
print(f"Prediction CPU Usage (Avg): {np.mean(prediction_cpu_usage):.2f}%")

# RAM 및 CPU 사용량 확인 (예측 후)
print("After Prediction:")
print_ram_usage()
print_cpu_usage()

# 예측값 생성 및 복원
y_pred = model.predict(X_test)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)
y_test_rescaled = target_scaler.inverse_transform(y_test)

# 예측값과 실제값 비교 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled.flatten(), label='Actual Close Values', marker='o')
plt.plot(y_pred_rescaled.flatten(), label='Predicted Close Values', marker='x')
plt.title('Actual vs Predicted Close Values')
plt.xlabel('Test Sample Index')
plt.ylabel('Close Value')
plt.legend()
plt.show()

# 최종 RAM 및 CPU 사용량 확인
print("Final Usage:")
print_ram_usage()
print_cpu_usage()
