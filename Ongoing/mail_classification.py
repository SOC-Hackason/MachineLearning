import torch
import pickle
from transformers import BertTokenizer, BertModel

# BERTモデルとトークナイザーの読み込み
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

#モデルを読み込む
def load_model(model_filename,scaler_filename):
   with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file) 
   with open(scaler_filename, 'rb') as file:
    loaded_scaler = pickle.load(file) 
    return loaded_model,loaded_scaler

#BERTの処理用関数  
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
        outputs = bert_model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
    return outputs.last_hidden_state[:, 0, :].numpy().squeeze()

#メール分類の確率分布を取得する関数
#メールの内容に加えて、ロードしたモデルとスケーラーが引数
def classify_mail(message,loaded_model,loaded_scaler):
    # 特徴量の抽出
    features = get_bert_features(message)
    
    # 特徴量の標準化
    features_scaled = loaded_scaler.transform([features])
    
    # クラスの確率を予測
    predicted_proba = loaded_model.predict_proba(features_scaled)
    
    return predicted_proba[0]