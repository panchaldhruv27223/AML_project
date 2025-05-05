import pandas as pd

csv_path = r"D:\Manish Prajapati\LibriSpeech\csv\test_split.csv"

data = pd.read_csv(csv_path)

print(len(data))

print(data.columns)