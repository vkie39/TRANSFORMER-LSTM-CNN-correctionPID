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
data = pd.read_csv("sensorData/merged_output_2025_01_02_23_07.csv")
data = data.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

# 입력과 출력 변수 분리
features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']
target = ['kp', 'ki', 'kd']

# 변화량 계산
data['delta_speed'] = data['target_speed'].diff().fillna(0)
data['delta_kp'] = data['kp'].diff().fillna(0)
data['delta_ki'] = data['ki'].diff().fillna(0)
data['delta_kd'] = data['kd'].diff().fillna(0)

# 변화량 임계값 설정
speed_threshold = 0.1
pid_threshold = 0.1

# 변화량 기준 데이터 필터링
filtered_data = data[
    (data['delta_speed'].abs() > speed_threshold) |
    (data['delta_kp'].abs() > pid_threshold) |
    (data['delta_ki'].abs() > pid_threshold) |
    (data['delta_kd'].abs() > pid_threshold)
]

print(f"Original Data Size: {len(data)}, Filtered Data Size: {len(filtered_data)}")

# 전체 데이터에서 입력과 출력 분리
X_full = data[features].values[:-1]
y_full = data[target].values[1:]

# 필터링된 데이터에서 입력과 출력 분리
X_filtered = filtered_data[features].values[:-1]
y_filtered = filtered_data[target].values[1:]

# 데이터 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X_full = input_scaler.fit_transform(X_full)
y_full = target_scaler.fit_transform(y_full)

X_filtered = input_scaler.transform(X_filtered)
y_filtered = target_scaler.transform(y_filtered)

scaler_dir = "LSTM_scalers"
os.makedirs(scaler_dir, exist_ok=True)

input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

joblib.dump(input_scaler, input_scaler_path)
joblib.dump(target_scaler, target_scaler_path)
print(f"Scalers saved to {scaler_dir}")

# Sliding Window를 이용한 시퀀스 데이터 생성
def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

seq_length = 5

# 전체 데이터 시퀀스 생성
X_seq_full, y_seq_full = create_sequences_sliding_window(X_full, y_full, seq_length)

# 필터링된 데이터 시퀀스 생성
X_seq_filtered, y_seq_filtered = create_sequences_sliding_window(X_filtered, y_filtered, seq_length)

# 훈련 데이터와 테스트 데이터 분리 (전체 데이터)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_seq_full, y_seq_full, test_size=0.2, shuffle=False
)

# 훈련 데이터와 테스트 데이터 분리 (필터링된 데이터)
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    X_seq_filtered, y_seq_filtered, test_size=0.2, shuffle=False
)

# 모델 생성 및 컴파일
model = create_model(lstm_units=64, dense_units=32, dropout_rate=0.2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="mse", metrics=["mae"])

# RAM 및 CPU 사용량 확인 (학습 전)
print("Before Training:")
print_ram_usage()
print_cpu_usage()

# 전체 데이터로 초기 학습
print("Training on full dataset...")
history_full = model.fit(
    X_train_full, y_train_full,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    shuffle=False
)

# 필터링된 데이터로 추가 학습
print("Fine-tuning on filtered dataset...")
history_filtered = model.fit(
    X_train_filtered, y_train_filtered,
    validation_split=0.1,
    epochs=50,
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

# 모델 평가 (전체 데이터)
print("Evaluating on full dataset...")
loss_full, mae_full = model.evaluate(X_test_full, y_test_full)
print(f"Test Loss (MSE) on Full Dataset: {loss_full}")
print(f"Test MAE on Full Dataset: {mae_full}")

# 모델 평가 (필터링된 데이터)
print("Evaluating on filtered dataset...")
loss_filtered, mae_filtered = model.evaluate(X_test_filtered, y_test_filtered)
print(f"Test Loss (MSE) on Filtered Dataset: {loss_filtered}")
print(f"Test MAE on Filtered Dataset: {mae_filtered}")

# 예측값 복원 및 비교 시각화
y_pred = model.predict(X_test_filtered)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)
y_test_rescaled = target_scaler.inverse_transform(y_test_filtered)

plt.figure(figsize=(10, 6))
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
