import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pid_LSTM_NN import create_model  # 모델 및 스케일러 가져오기
import joblib
import os

# 저장된 모델 경로
model_save_path = "model/saved_LSTM_model.keras"
model = tf.keras.models.load_model(model_save_path)

# 스케일러 로드
scaler_dir = "LSTM_scalers"
input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

input_scaler = joblib.load(input_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("Scalers loaded successfully")

# 속도 변화량 계산
def calculate_speed_change(recent_inputs):
    if len(recent_inputs) < 2:
        return 0
    return abs(recent_inputs[-1][0] - recent_inputs[-2][0])  # target_speed 변화량

class RealTimePredictor:
    def __init__(self, model, input_scaler, target_scaler, seq_length=5, threshold=0.05):
        self.model = model
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.recent_inputs = []  # 최근 입력값 저장소
        self.threshold = threshold  # 속도 변화 임계값

    def preprocess_input(self, raw_input):
        scaled_input = self.input_scaler.transform([raw_input])
        return scaled_input.flatten()

    def update_recent_inputs(self, input_features):
        self.recent_inputs.append(input_features)
        if len(self.recent_inputs) > self.seq_length:
            self.recent_inputs.pop(0)

    def predict_target(self):
        if len(self.recent_inputs) < self.seq_length:
            print(f"Not enough data to form a sequence. Waiting for {self.seq_length - len(self.recent_inputs)} more inputs.")
            return None

        speed_change = calculate_speed_change(self.recent_inputs)
       # if speed_change < self.threshold:
        #    print("Speed change below threshold. Skipping prediction.")
         #   return None

        input_sequence = np.expand_dims(np.array(self.recent_inputs), axis=0)
        scaled_prediction = self.model.predict(input_sequence)
        prediction = self.target_scaler.inverse_transform(scaled_prediction)
        return prediction.flatten()

    def run_real_time_prediction(self):
        print("Real-Time Prediction Started. Enter input features (comma-separated).")
        print("Input should be in the format: target_speed, cmd_vel_linear_x, pitch, mass (e.g., 1.2, 0.8, -0.1, 50)")

        while True:
            try:
                raw_input = input("Enter features (target_speed, cmd_vel_linear_x, pitch, mass): ").strip()
                if raw_input.lower() == "exit":
                    print("Exiting prediction system.")
                    break

                input_features = np.array([float(x) for x in raw_input.split(",")])
                scaled_features = self.preprocess_input(input_features)

                self.update_recent_inputs(scaled_features)

                predicted_target = self.predict_target()
                if predicted_target is not None:
                    print(f"Predicted Target: {predicted_target}")
            except ValueError:
                print("Invalid input. Please enter numeric values separated by commas.")
            except Exception as e:
                print(f"Error: {e}")

    def run_real_time_prediction_from_csv(self, csv_file):
        print(f"Starting prediction using CSV file: {csv_file}")
        try:
            data = pd.read_csv(csv_file)
            features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']
            inputs = data[features].values

            for raw_input in inputs:
                scaled_features = self.preprocess_input(raw_input)

                self.update_recent_inputs(scaled_features)

                predicted_target = self.predict_target()
                if predicted_target is not None:
                    print(f"Predicted Target (kp, ki, kd): {predicted_target}")
        except ValueError:
            print("Invalid input. Please enter numeric values separated by commas (e.g., 1.2, 0.8, -0.1, 50).")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    predictor = RealTimePredictor(model=model, input_scaler=input_scaler, target_scaler=target_scaler)
    mode = input("Choose prediction mode: [1] Real-Time, [2] CSV: ")
    if mode == "1":
        predictor.run_real_time_prediction()
    elif mode == "2":
        csv_path = input("Enter CSV file path: ")
        predictor.run_real_time_prediction_from_csv(csv_path)
    else:
        print("Invalid choice. Exiting.")
