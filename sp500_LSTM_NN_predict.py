import numpy as np
from sp500_LSTM_NN import input_scaler, target_scaler, model  # 스케일러와 모델 import

class RealTimePredictor:
    def __init__(self, model, input_scaler, target_scaler, seq_length=5, feature_window=25, threshold=0.3):
        """
        실시간 예측 초기화:
        - 모델, 스케일러, 시퀀스 길이, 이전 입력값 창 크기(feature_window), 검증 임계값 설정
        """
        self.model = model
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler
        self.seq_length = seq_length
        self.recent_inputs = []  # 최근 입력값 저장

        self.feature_window = feature_window  # 평균 비교를 위한 입력값 창 크기
        self.feature_window_inputs = []  # 입력값 평균 비교를 위한 저장소
        self.threshold = threshold  # 입력값 변화 검증 임계값

    def preprocess_input(self, raw_input):
        """
        입력 데이터를 스케일링 처리: 원본 데이터를 스케일링.
        """
        scaled_input = self.input_scaler.transform([raw_input])
        return scaled_input.flatten()  # 1D 배열로 반환

    def update_recent_inputs(self, input_features):
        """
        최근 입력값 업데이트:
        - 새로운 입력값을 최근 입력 리스트에 추가
        - 리스트 크기를 seq_length로 유지하며, 초과된 값은 제거
        """
        self.recent_inputs.append(input_features)
        if len(self.recent_inputs) > self.seq_length:
            self.recent_inputs.pop(0)

    def update_feature_window(self, input_features):
        """
        평균 비교를 위해 이전 입력값 저장 및 창 크기 유지
        """
        self.feature_window_inputs.append(input_features)
        if len(self.feature_window_inputs) > self.feature_window:
            self.feature_window_inputs.pop(0)

    def validate_prediction(self, current_input):
        """
        예측 값 검증:
        - 현재 예측에 사용된 입력값(5개)을 제외한 최근 20개 평균과 비교.
        - 매개변수로 현재 입력값(current_input)을 받아 검증 수행.
        """
        if len(self.feature_window_inputs) < self.feature_window:
            print("Not enough historical data to validate prediction. Skipping validation.")
            return True  # 초기 상태에서는 검증 없이 진행

        # 평균 계산 및 현재 입력값(5개)과 차이 비교
        feature_window_array = np.array(self.feature_window_inputs)
        data_for_mean = feature_window_array[:-self.seq_length]
        recent_mean = np.mean(data_for_mean, axis=0)
        recent_inputs_array = np.array(self.recent_inputs)

        # 각 입력값과 평균의 차이 계산
        diff = np.abs(recent_inputs_array - recent_mean)
        if np.all(diff < self.threshold):
            print("Prediction validated: Current features align with recent trends.")
            return False  # 유효하면 False 반환
        else:
            print("Prediction warning: Current features deviate significantly from recent trends.")
            print(f"Recent Mean (excluding current inputs): {recent_mean}")
            print(f"Recent Inputs: {recent_inputs_array}")
            print(f"Differences: {diff}")
            return True  # 이상이 있으면 True 반환

    def predict_target(self):
        """
        타겟 값 예측:
        - 최근 입력값이 seq_length에 도달했을 때 모델에 전달하여 타겟 값 예측
        """
        if len(self.recent_inputs) < self.seq_length:
            print(f"Not enough data to form a sequence. Waiting for {self.seq_length - len(self.recent_inputs)} more inputs.")
            return None

        # 시퀀스 생성 및 모델 입력
        input_sequence = np.array(self.recent_inputs)
        input_sequence = np.expand_dims(input_sequence, axis=0)  # (1, seq_length, features) 형태
        scaled_prediction = self.model.predict(input_sequence)
        prediction = self.target_scaler.inverse_transform(scaled_prediction)
        return prediction.flatten()[0]  # 예측값 반환

    def run_real_time_prediction(self):
        """
        실시간으로 입력 데이터를 받아 예측 수행
        """
        print("Real-Time Prediction Started. Enter input features (comma-separated).")
        print(f"Input should be in the format: Open, High, Low (e.g., 4300, 4350, 4280)")

        while True:
            try:
                raw_input = input("Enter features (Open, High, Low): ").strip()
                if raw_input.lower() == "exit":
                    print("Exiting prediction system.")
                    break

                # 사용자 입력 처리
                input_features = np.array([float(x) for x in raw_input.split(",")])
                scaled_features = self.preprocess_input(input_features)

                # 최근 입력값 업데이트
                self.update_recent_inputs(scaled_features)
                self.update_feature_window(scaled_features)

                # 타겟 예측
                predicted_target = self.predict_target()
                if predicted_target is not None:
                    print(f"Predicted Target: {predicted_target:.2f}")

                    # 예측 검증
                    is_valid = self.validate_prediction(scaled_features)
                    if not is_valid:
                        print("Warning: Significant deviation detected. Please check input data.")
            except ValueError:
                print("Invalid input. Please enter numeric values separated by commas (e.g., 4300, 4350, 4280).")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    # sp500_LSTM_NN.py에서 정의된 Scaler와 모델을 가져와 사용
    predictor = RealTimePredictor(model=model, input_scaler=input_scaler, target_scaler=target_scaler)
    predictor.run_real_time_prediction()
