import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  # 훈련/검증 데이터를 나누기 위한 라이브러리
from sklearn.preprocessing import MinMaxScaler       # 정규화를 위한 라이브러리

import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# CSV 파일 로드 (데이터 경로 수정 필요)
data = pd.read_csv("motor_data.csv")
data = data.replace(',', '', regex=True).astype(float)  # 쉼표 제거 및 숫자 변환

# 데이터 확인
print(data.head())

# 입력과 출력 변수 분리
features = ['target_speed', 'current_speed', 'voltage', 'current', 'motor_temp', 'ambient_temp']
target = ['p_value', 'i_value', 'd_value']  # PID 값을 개별 값으로 분리

X = data[features].values
y_pid = data[target].values  # 다중 출력 (p, i, d)

# 데이터 정규화 (-1 ~ 1 범위로)
scaler_X = MinMaxScaler(feature_range=(-1, 1))  # 입력값 정규화
scaler_y_pid = MinMaxScaler(feature_range=(-1, 1))  # PID 값 정규화

X = scaler_X.fit_transform(X)
y_pid = scaler_y_pid.fit_transform(y_pid)

# Sliding Window를 이용한 시퀀스 데이터 생성 함수
def create_sequences_sliding_window(X, y_pid, seq_length):
    X_seq, y_pid_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_pid_seq.append(y_pid[i + seq_length - 1])  # 마지막 시퀀스 값으로 PID 값 설정
    return np.array(X_seq), np.array(y_pid_seq)

seq_length = 50  # 시퀀스 길이

X_seq, y_pid_seq = create_sequences_sliding_window(X, y_pid, seq_length)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_pid_train, y_pid_test = train_test_split(
    X_seq, y_pid_seq, test_size=0.2, shuffle=False
)

# Transformer 블록 정의
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)  # 모든 입력 시퀀스간의 관계 학습
        self.ffn = tf.keras.Sequential([   # 활성화 함수를 통한 비선형 변환
            layers.Dense(ff_dim, activation="tanh"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)  # 모델 내부 특정 레이어의 출력값을 정규화함
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)  # 레이어를 지나며 잔차 연결로 인해 값이 너무 커지거나 작아질 수 있음
        self.dropout1 = layers.Dropout(dropout)  # 과적합 방지. 랜덤으로 일부 노드 비활성화
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)  # 입력 데이터의 관계 학습
        attn_output = self.dropout1(attn_output, training=training)  # 위 결과에 대한 Dropout 수행
        out1 = self.layernorm1(inputs + attn_output)  # 잔차 연결, 정규화
        ffn_output = self.ffn(out1)  # 비선형 변환
        ffn_output = self.dropout2(ffn_output, training=training)  # 위 결과에 Dropout 적용
        return self.layernorm2(out1 + ffn_output)  # 잔차 연결, 정규화

# Transformer + CNN 결합 모델 정의
class CNNSelfAttentionModel(tf.keras.Model):
    def __init__(self, seq_length, d_model, num_heads, ff_dim, cnn_filters):
        super(CNNSelfAttentionModel, self).__init__()
        self.conv1d = layers.Conv1D(filters=cnn_filters, kernel_size=3, activation="tanh", padding="same")  # 출력 데이터와 입력 데이터의 길이를 동일하게 유지함
        self.transformer = TransformerBlock(d_model, num_heads, ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc_pid = layers.Dense(3, activation="linear")  # 다중 출력 (p, i, d)를 위한 출력층

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)  # 1D CNN으로 로컬 패턴 추출
        x = self.transformer(x, training=training)  # Transformer로 시계열 학습
        x = self.global_pool(x)  # 전역 평균 풀링
        pid_output = self.fc_pid(x)  # PID 값 출력
        return pid_output  # 다중 출력 (p, i, d)

# 모델 초기화
seq_length = X_train.shape[1]
input_dim = X_train.shape[2]

model = CNNSelfAttentionModel(seq_length=seq_length, d_model=64, num_heads=4, ff_dim=128, cnn_filters=32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse",  # 다중 출력 회귀 문제
              metrics=["mae"])

history = model.fit(X_train, y_pid_train, 
                    validation_split=0.1, 
                    epochs=50, 
                    batch_size=32, 
                    shuffle=True)

# 테스트 데이터 평가
loss, mae = model.evaluate(X_test, y_pid_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# 테스트 데이터 예측
y_pid_pred = model.predict(X_test)
y_pid_pred_rescaled = scaler_y_pid.inverse_transform(y_pid_pred)  # PID 값 복원
y_pid_test_rescaled = scaler_y_pid.inverse_transform(y_pid_test)  # 테스트 PID 값 복원

# PID 값 출력
print("Predicted PID Values:")
print(y_pid_pred_rescaled)
print("\nActual PID Values:")
print(y_pid_test_rescaled)

# PID 값 시각화 (P, I, D 각각 비교)
plt.figure(figsize=(12, 8))
for i, label in enumerate(['P', 'I', 'D']):
    plt.subplot(3, 1, i+1)
    plt.plot(y_pid_test_rescaled[:, i], label=f'Actual {label}', marker='o')
    plt.plot(y_pid_pred_rescaled[:, i], label=f'Predicted {label}', marker='x')
    plt.title(f'Actual vs Predicted {label} Values')
    plt.xlabel('Test Sample Index')
    plt.ylabel(f'{label} Value')
    plt.legend()
plt.tight_layout()
plt.show()


#===================================================================
# 실시간 예측
'''
def predict_real_time(input_data):
    # 저장된 Scaler 로드
    scaler = joblib.load("scaler.pkl")
    model = tf.keras.models.load_model("cnn_transformer_model.h5", custom_objects={"TransformerBlock": TransformerBlock})
    
    # 입력 데이터 스케일링
    input_data_scaled = scaler.transform(input_data)
    
    # 시퀀스 형태로 변환
    input_seq = np.expand_dims(input_data_scaled, axis=0)  # (1, seq_length, feature_dim) 형태
    
    # 모델 예측
    prediction = model.predict(input_seq)
    return prediction

# 실시간 예측 테스트
new_input = np.array([[50, 40, 220, 10, 75, 25]])  # 새로운 입력 데이터
predicted_pid_values = predict_real_time(new_input)
print(f"Predicted P: {predicted_pid_values[0][0]:.4f}")
print(f"Predicted I: {predicted_pid_values[0][1]:.4f}")
print(f"Predicted D: {predicted_pid_values[0][2]:.4f}")

# 학습 결과 시각화
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''
