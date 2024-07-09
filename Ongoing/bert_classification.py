import pandas as pd
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertModel

# Load dataset
df = pd.read_csv('maildata.csv') 

# Split the data into features and labels
mail_text = df['text']
y = df['label']


# BERTモデルの準備
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

def get_bert_features(text, max_length=128):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
    return outputs.last_hidden_state[:, 0, :].numpy().squeeze()

features = mail_text.apply(get_bert_features)
X = np.stack(features.values)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化のためのスケーラーを適用
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ロジスティック回帰モデルのトレーニング
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# モデルの評価
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred,zero_division=1))
#appontmment,promotion,school,work
print('Predicted Probabilities:\n', y_pred_proba)
print(y_pred)

with open('model.pickle', mode='wb') as f:
    pickle.dump(model,f,protocol=2)

with open('scaler.pickle', mode='wb') as f:
    pickle.dump(scaler,f)