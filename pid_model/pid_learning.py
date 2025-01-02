import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pid_transformer_cnn import create_model
from tensorflow.keras import layers
import pandas as pd
import joblib

# CSV 파일 로드 및 데이터 전처리
data = pd.read_csv("sensorData/merged_output_2025_01_02_23_07.csv")
data = data.replace(',', '', regex=True).astype(float)

features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']  # 입력 변수
target = ['kp', 'ki', 'kd']  # 출력 변수

# 변화량 계산
data['delta_speed'] = data['target_speed'].diff().fillna(0)
data['delta_kp'] = data['kp'].diff().fillna(0)
data['delta_ki'] = data['ki'].diff().fillna(0)
data['delta_kd'] = data['kd'].diff().fillna(0)

# 변화량 임계값 설정
speed_threshold = 0.1
pid_threshold = 0.1

# 변화량 기준 데이터 필터링 (변화가 있는 부분에 대한 추가학습을 하기 위함함)
filtered_data = data[
    (data['delta_speed'].abs() > speed_threshold) |
    (data['delta_kp'].abs() > pid_threshold) |
    (data['delta_ki'].abs() > pid_threshold) |
    (data['delta_kd'].abs() > pid_threshold)
]
print(f"Original Data Size: {len(data)}, Filtered Data Size: {len(filtered_data)}")

#원본 데이터터
X = data[features].values[:-1]
y = data[target].values[1:]

# 필터링된 데이터에서 입력과 출력 분리
X_filtered = filtered_data[features].values[:-1]
y_filtered = filtered_data[target].values[1:]

# 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = target_scaler.fit_transform(y)

X_filtered = input_scaler.transform(X_filtered)
y_filtered = target_scaler.transform(y_filtered)

scaler_dir = "transformer_scalers"
os.makedirs(scaler_dir, exist_ok=True)

input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

joblib.dump(input_scaler, input_scaler_path)
joblib.dump(target_scaler, target_scaler_path)
print(f"Scalers saved to {scaler_dir}")


seq_length = 5  # 시퀀스 길이


def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_sliding_window(X, y, seq_length)
X_seq_filtered, y_seq_filtered = create_sequences_sliding_window(X_filtered, y_filtered, seq_length)


# 데이터 및 스케일러 저장
#def get_data():
#    return X_seq, y_seq, input_scaler, target_scaler


# 데이터 로드
#X_seq, y_seq, input_scaler, target_scaler = get_data()

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    X_seq_filtered, y_seq_filtered, test_size=0.2, shuffle=False
)

# 모델 생성
seq_length = X_train.shape[1]
input_dim = X_train.shape[2]
output_dim = y_train.shape[1]

model = create_model(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss="mse",
              metrics=["mae"])


# 검증 손실(val_loss) 개선 안될 시 동적으로 학습률 조정
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=1e-6)

# 모델 학습
print("Training on full dataset...")
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=32,
                    shuffle=False,
                    callbacks=[reduce_lr]
                    )

# 필터링된 데이터로 추가 학습
print("Fine-tuning on filtered dataset...")
history_filtered = model.fit(
    X_train_filtered, y_train_filtered,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    shuffle=False,
    callbacks=[reduce_lr]
)

# 모델 경로 생성
model_save_dir = "model"
os.makedirs(model_save_dir, exist_ok=True)  # 폴더 생성 (이미 존재해도 에러 없음)
model_save_path = os.path.join(model_save_dir, "saved_transformer_model.keras")
# 모델 저장 (model 폴더 아래로)
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 모델 평가 (전체 데이터)
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# 모델 평가 (필터링된 데이터)
print("Evaluating on filtered dataset...")
loss_filtered, mae_filtered = model.evaluate(X_test_filtered, y_test_filtered)
print(f"Test Loss (MSE) on Filtered Dataset: {loss_filtered}")
print(f"Test MAE on Filtered Dataset: {mae_filtered}")


# 예측값 생성 및 시각화
y_pred = model.predict(X_test)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)
y_test_rescaled = target_scaler.inverse_transform(y_test)

# 다중 출력 시각화
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
