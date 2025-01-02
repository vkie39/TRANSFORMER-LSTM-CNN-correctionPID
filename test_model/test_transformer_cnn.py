import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time
import psutil

# CSV 파일 로드
data = pd.read_csv("sample_data_sp500.csv")
data = data.replace(',', '', regex=True).astype(float)

# 입력과 출력 변수 분리
features = ['Open', 'High', 'Low']
target = 'Close'

X = data[features].values  # (samples, features)
y = data[target].values  # (samples,)

# 스케일링
input_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler = MinMaxScaler(feature_range=(-1, 1))

X = input_scaler.fit_transform(X)
y = target_scaler.fit_transform(y.reshape(-1, 1))

seq_length = 5  # 시퀀스 길이

# Sliding Window를 이용한 시퀀스 데이터 생성
def create_sequences_sliding_window(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences_sliding_window(X, y, seq_length)

# 데이터 차원 확인
print(f"X_seq shape: {X_seq.shape}")  # (num_sequences, seq_length, features)
print(f"y_seq shape: {y_seq.shape}")  # (num_sequences, 1)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Transformer 블록 정의
class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="tanh"),
            layers.Dense(d_model)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer + CNN 결합 모델 정의
class CNNSelfAttentionModel(tf.keras.Model):
    def __init__(self, seq_length, d_model, num_heads, ff_dim, cnn_filters):
        super(CNNSelfAttentionModel, self).__init__()
        self.conv1d = layers.Conv1D(filters=cnn_filters, kernel_size=3, activation="tanh", padding="same")
        self.transformer = TransformerBlock(d_model, num_heads, ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(1, activation="linear")

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.transformer(x, training=training)
        x = self.global_pool(x)
        return self.fc(x)

# 모델 초기화
seq_length = X_train.shape[1]
input_dim = X_train.shape[2]

model = CNNSelfAttentionModel(seq_length=seq_length, d_model=32, num_heads=4, ff_dim=128, cnn_filters=32)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss="mse",
              metrics=["mae"])

# 모델 학습
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=100,
                    batch_size=32,
                    shuffle=False)

# 모델 평가
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss (MSE): {loss}")
print(f"Test MAE: {mae}")

# 예측값 생성 및 시각화
y_pred = model.predict(X_test)
y_pred_rescaled = target_scaler.inverse_transform(y_pred)
y_test_rescaled = target_scaler.inverse_transform(y_test)

plt.figure(figsize=(10, 6))
plt.plot(y_test_rescaled.flatten(), label='Actual Close Values', marker='o')
plt.plot(y_pred_rescaled.flatten(), label='Predicted Close Values', marker='x')
plt.title('Actual vs Predicted Close Values')
plt.xlabel('Test Sample Index')
plt.ylabel('Close Value')
plt.legend()
plt.show()
