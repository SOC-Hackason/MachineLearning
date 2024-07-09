import os
import pandas as pd

# ディレクトリ内のCSVファイルを取得
directory = 'Dataset'
all_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.csv')]

# CSVファイルを1つのデータフレームに結合する
combined_df = pd.concat([pd.read_csv(file) for file in all_files], ignore_index=True)

# 結合したデータをCSVファイルとして保存
combined_df.to_csv('maildata.csv', index=False)