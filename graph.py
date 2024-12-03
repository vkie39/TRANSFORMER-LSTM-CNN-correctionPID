import pandas as pd
import matplotlib.pyplot as plt

file_path = 'sample_data_sp500.csv'

try:
    df = pd.read_csv(file_path)
    df = df.replace(',', '', regex=True).astype(float)

    if 'Open' in df.columns:
        open_col = df['Open']

        plt.figure(figsize=(10, 5))
        plt.plot(open_col, marker='o', label='Open')
        plt.title('Graph of Open Column')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("'Open' 열이 존재하지 않습니다.")
except FileNotFoundError:
    print(f"'{file_path}' 파일이 존재하지 않습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
