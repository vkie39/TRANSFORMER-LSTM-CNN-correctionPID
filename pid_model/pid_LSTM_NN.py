import tensorflow as tf
from tensorflow.keras import layers

# LSTM + Fully Connected Neural Network (NN) 모델 정의
class LSTMNNModel(tf.keras.Model):
    def __init__(self, lstm_units, dense_units, dropout_rate=0.1):
        super(LSTMNNModel, self).__init__()
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
        self.lstm = layers.LSTM(units=lstm_units, return_sequences=False)  # LSTM 레이어
        self.dropout = layers.Dropout(dropout_rate)  # Dropout 레이어
        self.dense1 = layers.Dense(units=dense_units, activation="linear")  # Fully Connected 레이어
        self.dense2 = layers.Dense(3, activation="linear")  # 출력 레이어

    def call(self, inputs, training=False):
        x = self.lstm(inputs, training=training)  # LSTM 처리
        x = self.dropout(x, training=training)   # Dropout 적용
        x = self.dense1(x)                       # 첫 번째 Dense 레이어
        return self.dense2(x)                    # 출력 레이어
    

    def get_config(self):
        """
        모델의 구성 정보를 반환하는 메서드.
        """
        # config = super(LSTMNNModel, self).get_config()
        # config.update({
        #     'lstm_units': self.lstm_units,
        #     'dense_units': self.dense_units,
        #     'dropout_rate': self.dropout_rate,
        # })
        config = {
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
        }
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        구성 정보를 사용하여 모델을 복원하는 클래스 메서드.
        """
        return cls(**config)

# 모델 생성 함수

def create_model(lstm_units=64, dense_units=32, dropout_rate=0.2):
    """
    모델 생성 함수로, 외부에서 lstm_units, dense_units, dropout_rate를 설정 가능.
    Default 값: lstm_units=64, dense_units=32, dropout_rate=0.2.
    
    :param lstm_units: LSTM 노드 수
    :param dense_units: Dense 노드 수
    :param dropout_rate: Dropout 비율
    :return: 초기화된 LSTMNNModel 인스턴스
    """
    return LSTMNNModel(lstm_units=lstm_units, dense_units=dense_units, dropout_rate=dropout_rate)
