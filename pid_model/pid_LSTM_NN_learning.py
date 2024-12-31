import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import psutil
import time
import os
from pid_LSTM_NN import create_model  # 모델 생성 함수 불러오기
import joblib

# RAM 사용량 확인 함수
def print_ram_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    rss_memory = memory_info.rss / (1024 ** 2)  # Resident Set Size (MB)
    print(f"RAM Usage: {rss_memory:.2f} MB")

# CPU 사용량 확인 함수
def print_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)  # 1초 간격으로 CPU 사용량 측정
    print(f"CPU Usage: {cpu_percent:.2f}%")

# CSV 파일 로드
data = pd.read_csv("sensorData/merged_output_2024_12_31_08_02.csv")
data = data.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

# 입력과 출력 변수 분리
features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']
target = ['kp', 'ki', 'kd']

X = data[features].values[:-1]
y = data[target].values[1:]

# 데이터 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = target_scaler.fit_transform(y)

scaler_dir = "LSTM_scalers"
os.makedirs(scaler_dir, exist_ok=True)

input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

joblib.dump(input_scaler, input_scaler_path)
joblib.dump(target_scaler, target_scaler_path)
print(f"Scalers saved to {scaler_dir}")

# Sliding Window를 이용한 시퀀스 데이터 생성
seq_length = 5

def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_sliding_window(X, y, seq_length)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# 모델 생성 및 컴파일
model = create_model(lstm_units=64, dense_units=32, dropout_rate=0.2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# RAM 및 CPU 사용량 확인 (학습 전)
print("Before Training:")
print_ram_usage()
print_cpu_usage()

# 모델 학습
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    shuffle=False
)

# 모델 저장 (model 폴더 아래로)
model_save_dir = "model"
os.makedirs(model_save_dir, exist_ok=True)  # 폴더 생성 (이미 존재해도 에러 없음)
model_save_path = os.path.join(model_save_dir, "saved_LSTM_model.keras")  # .keras 확장자 추가
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# RAM 및 CPU 사용량 확인 (학습 후)
print("After Training:")
print_ram_usage()
print_cpu_usage()

# 모델 평가
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# 예측값 복원 및 비교 시각화
y_pred = model.predict(X_test)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)
y_test_rescaled = target_scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
target = ['kp', 'ki', 'kd']  # 출력 변수
for i, label in enumerate(target):
    plt.plot(y_test_rescaled[:, i], label=f'Actual {label}', marker='o')
    plt.plot(y_pred_rescaled[:, i], label=f'Predicted {label}', marker='x')
plt.title('Actual vs Predicted PID Values')
plt.xlabel('Test Sample Index')
plt.ylabel('PID Values')
plt.legend()
plt.show()

# 최종 리소스 사용량 확인
print("Final Usage:")
print_ram_usage()
print_cpu_usage()
