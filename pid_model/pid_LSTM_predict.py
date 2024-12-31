import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from pid_LSTM_NN import create_model  # Scaler와 모델 가져오기
import joblib
import os

# 저장된 모델 경로
model_save_path = "model/saved_LSTM_model.keras"
model = tf.keras.models.load_model(model_save_path)

# 스케일러 초기화
# input_scaler = MinMaxScaler(feature_range=(-1, 1))  # 입력 데이터 범위에 맞게 초기화
# target_scaler = MinMaxScaler(feature_range=(-1, 1))  # 출력 데이터 범위에 맞게 초기화
scaler_dir = "LSTM_scalers"
input_scaler_path = os.path.join(scaler_dir, "input_scaler.pkl")
target_scaler_path = os.path.join(scaler_dir, "target_scaler.pkl")

input_scaler = joblib.load(input_scaler_path)
target_scaler = joblib.load(target_scaler_path)
print("Scalers loaded successfully")

class RealTimePredictor:
    def __init__(self, model, input_scaler, target_scaler, seq_length=5, feature_window=25, threshold1=0.3, threshold2=0.3, history_window=5):
        """
        실시간 예측 초기화:
        - 모델, 스케일러, 시퀀스 길이, 이전 입력값 창 크기, 검증 임계값 설정.
        """
        self.model = model
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.recent_inputs = []  # 최근 입력값 저장소
        self.feature_window = feature_window  # 평균 비교를 위한 입력값 창 크기
        self.feature_window_inputs = []  # 입력값 평균 비교 저장소
        self.threshold1 = threshold1  # 입력값 변화 검증 임계값
        self.threshold2 = threshold2  # 입력값 변화 검증 임계값
        self.predicted_targets = []  # 예측된 타겟값들을 저장할 리스트
        self.history_window = history_window  # 예측값 비교를 위한 기록 창 크기

    def preprocess_input(self, raw_input):
        """입력 데이터를 스케일링 처리"""
        scaled_input = self.input_scaler.transform([raw_input])
        return scaled_input.flatten()

    def update_recent_inputs(self, input_features):
        """최근 입력값 업데이트"""
        self.recent_inputs.append(input_features)
        if len(self.recent_inputs) > self.seq_length:
            self.recent_inputs.pop(0)

    def update_feature_window(self, input_features):
        """이전 입력값 창 업데이트"""
        self.feature_window_inputs.append(input_features)
        if len(self.feature_window_inputs) > self.feature_window:
            self.feature_window_inputs.pop(0)

    def validate_prediction(self):
        """예측 값 검증"""
        if len(self.feature_window_inputs) < self.feature_window:
            print("Not enough historical data to validate prediction.")
            return True

        feature_window_array = np.array(self.feature_window_inputs)
        data_for_mean = feature_window_array[:-self.seq_length]
        recent_mean = np.mean(data_for_mean, axis=0)
        recent_inputs_array = np.array(self.recent_inputs)

        diff = np.abs(recent_inputs_array - recent_mean)
        if np.all(diff < self.threshold1):
            print("Prediction validated: Current features align with recent trends.")
            return False
        else:
            print("Prediction warning: Significant deviation detected.")
            print(f"Recent Mean (excluding current inputs): {recent_mean}")
            print(f"Recent Inputs: {recent_inputs_array}")
            print(f"Differences: {diff}")
            return True

    def predict_target(self):
        """타겟 값 예측"""
        if len(self.recent_inputs) < self.seq_length:
            print(f"Not enough data to form a sequence. Waiting for {self.seq_length - len(self.recent_inputs)} more inputs.")
            return None

        input_sequence = np.expand_dims(np.array(self.recent_inputs), axis=0)
        scaled_prediction = self.model.predict(input_sequence)
        prediction = self.target_scaler.inverse_transform(scaled_prediction)
        return prediction.flatten()

    def run_real_time_prediction(self):
        """실시간 입력 데이터 처리 및 예측"""
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
                self.update_feature_window(scaled_features)

                predicted_target = self.predict_target()
                if predicted_target is not None:
                    print(f"Predicted Target: {predicted_target}")

                    if not self.validate_prediction():
                        print("Warning: Significant deviation detected. Please check input data.")
            except ValueError:
                print("Invalid input. Please enter numeric values separated by commas.")
            except Exception as e:
                print(f"Error: {e}")

    def run_real_time_prediction_from_csv(self, csv_file):
        """CSV 파일로부터 입력 데이터 처리 및 예측"""
        print(f"Starting prediction using CSV file: {csv_file}")
        try:
            data = pd.read_csv(csv_file)
            features = ['target_speed', 'cmd_vel_linear_x', 'pitch', 'mass']
            inputs = data[features].values

            for raw_input in inputs:
                scaled_features = self.preprocess_input(raw_input)

                self.update_recent_inputs(scaled_features)
                self.update_feature_window(scaled_features)

                predicted_target = self.predict_target()
                if predicted_target is not None:
                    print(f"Predicted Target (kp, ki, kd): {predicted_target}")

                    # 예측 검증
                    # 이전 예측값들과 비교 (가장 최근 예측을 제외한 4래 예측값들과의 차이 계산)
                    if len(self.predicted_targets) >= 5:
                        previous_predictions = np.array(self.predicted_targets[:-1])  # 가장 최근 값을 제외한 4개 값
                        previous_avg = np.mean(previous_predictions, axis=0)

                        diff = np.abs(predicted_target - previous_avg)
                        print(f"Difference from previous predictions: {diff}")

                        # 예측값 차이가 임계값을 초과하는지 확인
                        if np.any(diff > self.threshold2):
                            print("Significant difference detected. Performing validation.")
                            is_valid = self.validate_prediction(predicted_target)
                            if not is_valid:
                                print("Warning: Significant deviation detected. Please check input data.")
                        else:
                            print("Predicted value is similar to the previous values. Skipping validation.")
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
