import re
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import requests

# スパイシーの日本語モデルを読み込み
nlp = spacy.load('ja_ginza')

def preprocess_email(email_text):
    # HTMLタグを削除
    email_text = re.sub('<[^<]+?>', '', email_text)
    # 日本語の非文字を削除
    email_text = re.sub(r'[^\w\s]', '', email_text)
    # 小文字変換（日本語には無関係）
    email_text = email_text.lower()

    # 常用の挨拶と結辞の削除（日本語版）
    email_text = re.sub(r'\b(親愛なる|よろしくお願いします|お世話になります|こんにちは|こんばんは|ありがとう|感謝します|よろしく)\b[\s,]*', '', email_text)
    
    # 分かち書きと品詞タグ付け
    doc = nlp(email_text)
    # 形態素解析とストップワードの除去
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def extract_custom_keywords(original_text, preprocessed_text):
    # スパイシーでのエンティティ認識
    doc = nlp(original_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 特定のキーワードの抽出
    keywords = []
    for ent in entities:
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'DATE']:
            keywords.append(ent[0])
    
    # 特定のフレーズの検出
    phrases = ['ネコのミーム動画', '誕生日パーティ', '大丈夫']
    for phrase in phrases:
        if phrase in original_text:
            keywords.append(phrase)
    
    # 不要なフレーズの除去
    exclude_phrases = ['最近', 'ここ2週間']
    for phrase in exclude_phrases:
        preprocessed_text = preprocessed_text.replace(phrase, '')

    # 高頻度単語のカウントと除外
    exclude_words = ['これ', 'それ', 'こと', 'どう', 'わかる', '聞く', '見る', '言う', 'いる', '週間']
    word_freq = Counter(preprocessed_text.split())
    high_freq_words = [word for word, freq in word_freq.items() if freq > 1 and word not in phrases and len(word) > 1 and word not in exclude_words]

    keywords.extend(high_freq_words)
    
    # 重複の除去
    keywords = list(set(keywords))
    return keywords

def vectorize_keywords(keywords):
    vectorizer = TfidfVectorizer()
    keyword_vectors = vectorizer.fit_transform(keywords)
    return vectorizer, keyword_vectors

def classify_email(model, vectorizer, email_text):
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)
    return prediction[0]

def send_line_notification(message, user_id, access_token):
    url = 'https://api.line.me/v2/bot/message/push'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }
    data = {
        'to': user_id,
        'messages': [{'type': 'text', 'text': message}]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.status_code

# サンプルメールテキスト
email_text = """
こんにちは
私はスパム太郎です。
極めて胡乱なビジネスチャンスがあります！
"""

# メールの前処理
preprocessed_text = preprocess_email(email_text)
print("前処理後のテキスト：", preprocessed_text)

# カスタムキーワードの抽出
custom_keywords = extract_custom_keywords(email_text, preprocessed_text)
print("カスタムキーワード：", custom_keywords)

# サンプルデータセット
data = [
    {'text': '大丈夫 ネコのミーム動画 Prof. K 6月20日 誕生日パーティ', 'label': 'social'},
    {'text': 'プロジェクト 締め切り 6月20日 ミーティング', 'label': 'work'},
    # 他のサンプルを追加
]

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# キーワードのベクトル化
vectorizer, keyword_vectors = vectorize_keywords(texts)

# 訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(keyword_vectors, labels, test_size=0.2, random_state=42)

# MultinomialNBモデルの訓練
model = MultinomialNB()
model.fit(X_train, y_train)

# テストセットでの予測と評価
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("モデルの精度：", accuracy)

# 新しいメールの例
new_email_text = 'Prof. Kの誕生日パーティは6月20日です'
category = classify_email(model, vectorizer, new_email_text)
print("メールの分類結果：", category)

# LINE通知の送信
user_id = ''  # 提供されたJSONデータから取得
access_token = ''  # チャンネルアクセストークン
message = f'新しい {category} メールがあります: {new_email_text}'
status_code = send_line_notification(message, user_id, access_token)
print("通知のステータスコード：", status_code)
