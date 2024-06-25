import re
import spacy
from collections import Counter

# 日本語のspacyモデルをロード
nlp = spacy.load('ja_ginza')

def preprocess_email(email_text):
    # HTMLタグを除去
    email_text = re.sub('<[^<]+?>', '', email_text)
    # 非文字を除去して空白を残す
    email_text = re.sub(r'[^ぁ-んァ-ンーa-zA-Z一-龠０-９\s]', '', email_text)
    # 小文字に変換
    email_text = email_text.lower()

    # よく使う挨拶や締めの言葉を除去
    email_text = re.sub(r'\b(拝啓|敬具|お世話になります|よろしくお願いします|お疲れ様です|ありがとうございます|失礼します)\b[\s,]*', '', email_text)

    # 分かち書きと品詞タグ付け
    doc = nlp(email_text)
    # 形態素の基本形に変換し、ストップワードを除去
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def extract_custom_keywords(email_text):
    # spacyで命名エンティティを認識
    doc = nlp(email_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # 特定のキーワードを抽出
    keywords = []
    for ent in entities:
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'DATE']:
            keywords.append(ent[0])
    
    # 特定のフレーズを検索
    phrases = ['ねこミーム動画', '誕生日パーティー', 'ok']
    for phrase in phrases:
        if phrase in email_text.lower():
            keywords.append(phrase)

    # 不必要なフレーズを排除
    exclude_phrases = ['最近', 'この二週間']
    for phrase in exclude_phrases:
        email_text = email_text.replace(phrase, '')

    # 頻出単語をカウントして不要な単語を除外
    exclude_words = ['これら', 'あなた', 'あります', 'これ', 'それ', '知る', '書く', '問う', '見る', 'しましょう', '週']
    word_freq = Counter(email_text.split())
    high_freq_words = [word for word, freq in word_freq.items() if freq > 1 and word not in phrases and len(word) > 2 and word not in exclude_words]

    keywords.extend(high_freq_words)

    # 重複を除去
    keywords = list(set(keywords))
    return keywords

# サンプルメールテキスト
email_text = """
拝啓 メールソムリエ、

最近はいかがですか？この二週間の間に、何かご報告することはありますか？ねこミーム動画をいくつか見ました、こちらをご覧ください
https://x.com/haitekukaito/status/1754837346376438083

ところで、K教授の誕生日パーティーが6月20日にあります。興味があれば教えてください。お返事を楽しみにしています。

敬具

Qiu
"""

# メールテキストを前処理
preprocessed_text = preprocess_email(email_text)
print("前処理後のテキスト：", preprocessed_text)

# カスタムキーワードを抽出
custom_keywords = extract_custom_keywords(email_text)
print("カスタムキーワード：", custom_keywords)
