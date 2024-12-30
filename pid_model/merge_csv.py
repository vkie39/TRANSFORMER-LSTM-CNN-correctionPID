import os
import pandas as pd
from datetime import datetime

# Directory containing the CSV files
input_directory = "C://Users//82104//Documents//GitHub//TRANSFORMER-LSTM-CNN-correctionPID//sensorData"

# Generate timestamped output file name
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
output_file = f"C://Users//82104//Documents//GitHub//TRANSFORMER-LSTM-CNN-correctionPID//sensorData//merged_output_{timestamp}.csv"

# List to hold DataFrames
dataframes = []

# Iterate over all CSV files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(input_directory, filename)
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Append the DataFrame to the list
        dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Replace NaN values with 0
merged_df.fillna(0, inplace=True)

# Save the merged DataFrame to a CSV file
merged_df.to_csv(output_file, index=False)

print(f"Merged CSV file saved to {output_file}")