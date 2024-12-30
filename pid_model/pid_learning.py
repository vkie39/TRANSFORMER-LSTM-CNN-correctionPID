import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from pid_transformer_cnn import create_model
from tensorflow.keras import layers

# CSV 파일 로드 및 데이터 전처리
data = pd.read_csv("sample_data_sp500.csv")
data = data.replace(',', '', regex=True).astype(float)

features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']  # 입력 변수
target = ['kp', 'ki', 'kd']  # 출력 변수

X = data[features].values
y = data[target].values

# 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = target_scaler.fit_transform(y)

seq_length = 5  # 시퀀스 길이


def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_sliding_window(X, y, seq_length)

# 데이터 및 스케일러 저장
def get_data():
    return X_seq, y_seq, input_scaler, target_scaler


# 데이터 로드
X_seq, y_seq, input_scaler, target_scaler = get_data()

# 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# 모델 생성
seq_length = X_train.shape[1]
input_dim = X_train.shape[2]
output_dim = y_train.shape[1]

model = create_model(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length)

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss="mse",
              metrics=["mae"])

# 모델 학습
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=100,
                    batch_size=32,
                    shuffle=False)

# 모델 저장 (model 폴더 아래로)
model_save_dir = "model"
os.makedirs(model_save_dir, exist_ok=True)  # 폴더 생성 (이미 존재해도 에러 없음)
model_save_path = os.path.join(model_save_dir, "saved_pid_model")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# 모델 평가
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

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
