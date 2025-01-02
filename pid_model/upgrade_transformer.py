import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers

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

        # Config에 저장할 속성들
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout': self.dropout,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# Transformer 기반 모델 정의
class TransformerOnlyModel(tf.keras.Model):
    def __init__(self, seq_length, d_model, num_heads, ff_dim, output_dim):
        super(TransformerOnlyModel, self).__init__()
        self.transformer = TransformerBlock(d_model, num_heads, ff_dim)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.fc = layers.Dense(output_dim, activation="tanh")

        # Config에 저장할 속성들
        self.seq_length = seq_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.output_dim = output_dim

    def call(self, inputs, training=False):
        x = self.transformer(inputs, training=training)
        x = self.global_pool(x)
        return self.fc(x)

    def get_config(self):
        # super().get_config()를 호출하지 않음
        config = {
            'seq_length': self.seq_length,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'output_dim': self.output_dim,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 모델 생성 함수
def create_model(input_dim, output_dim, seq_length):
    return TransformerOnlyModel(seq_length=seq_length, d_model=32, num_heads=4, ff_dim=128, output_dim=output_dim)
