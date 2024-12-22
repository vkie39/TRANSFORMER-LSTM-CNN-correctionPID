import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
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
    def __init__(self, seq_length, d_model, num_heads, ff_dim, cnn_filters, output_dim):
        super(CNNSelfAttentionModel, self).__init__()
        self.conv1d = layers.Conv1D(filters=cnn_filters, kernel_size=3, activation="tanh", padding="same")
        self.transformer = TransformerBlock(d_model, num_heads, ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(output_dim, activation="linear")  # 다중 출력 (kp, ki, kd)를 위한 출력층

    def call(self, inputs, training=False):
        x = self.conv1d(inputs)
        x = self.transformer(x, training=training)
        x = self.global_pool(x)
        return self.fc(x)


# 데이터 및 스케일러 저장
def get_data():
    return X_seq, y_seq, input_scaler, target_scaler


# 모델 생성 함수
def create_model(input_dim, output_dim, seq_length):
    return CNNSelfAttentionModel(seq_length=seq_length, d_model=32, num_heads=4, ff_dim=128, cnn_filters=32, output_dim=output_dim)
