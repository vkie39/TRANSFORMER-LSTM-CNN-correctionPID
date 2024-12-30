import pandas as pd

filepath = "C://Users//82104//Documents//GitHub//TRANSFORMER-LSTM-CNN-correctionPID//sensorData//2024_12_30_15_53.csv"

df = pd.read_csv(filepath)
df_filled = df.fillna(0, inplace=True)
df.to_csv(filepath, index = False)