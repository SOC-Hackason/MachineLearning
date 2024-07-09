import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('spam_ham_dataset.csv')

# 特定の列の値が条件を満たす行を抽出する
filtered_df = df[df['label'] == 'spam']

# 条件を満たす最初の100行だけを取り出す
filtered_df = filtered_df.head(100)

# 左から2列だけを取り出す
result_df = filtered_df.iloc[:, 1:]

# 別のCSVファイルに書き出す
result_df.to_csv('output.csv', index=False)