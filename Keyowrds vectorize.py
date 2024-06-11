import re
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import requests

# 加载spacy模型
nlp = spacy.load('en_core_web_sm')

def preprocess_email(email_text):
    # 去除HTML标签
    email_text = re.sub('<[^<]+?>', '', email_text)
    # 去除非字母字符，保留空格
    email_text = re.sub(r'[^a-zA-Z\s]', '', email_text)
    # 转换为小写
    email_text = email_text.lower()

    # 去除常见称呼和结束语
    email_text = re.sub(r'\b(dear|regards|best regards|sincerely|hello|hi|thank you|thanks|yours|truly|faithfully|cheers|kind regards|mailsommelier)\b[\s,]*', '', email_text)
    
    # 分词与词性标注
    doc = nlp(email_text)
    # 词形还原并去除停用词
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def extract_custom_keywords(original_text, preprocessed_text):
    # 使用spacy进行命名实体识别
    doc = nlp(original_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 提取特定关键词
    keywords = []
    for ent in entities:
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'DATE']:
            keywords.append(ent[0])
    
    # 查找特定短语
    phrases = ['neko meme videos', 'birthday party', 'ok']
    for phrase in phrases:
        if phrase in original_text.lower():
            keywords.append(phrase)
    
    # 排除不必要的短语
    exclude_phrases = ['these days', 'these two weeks']
    for phrase in exclude_phrases:
        preprocessed_text = preprocessed_text.replace(phrase, '')

    # 统计高频词，排除不需要的词
    exclude_words = ['these', 'you', 'are', 'this', 'that', 'know', 'write', 'ask', 'see', 'let', 'm', 'week']
    word_freq = Counter(preprocessed_text.split())
    high_freq_words = [word for word, freq in word_freq.items() if freq > 1 and word not in phrases and len(word) > 2 and word not in exclude_words]

    keywords.extend(high_freq_words)
    
    # 去重
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

# 示例邮件文本
email_text = """
Dear mailsommelier,

I want to know if it's ok for you recently. I'm writing to ask if you are ok in these two weeks. I saw some neko meme videos these days, check this out
https://x.com/haitekukaito/status/1754837346376438083

However, Prof. K is having a birthday party on June.20th. If you are interested, please let me know. Looking forward to hearing from you.

Best regards,

Qiu
"""

# 预处理邮件文本
preprocessed_text = preprocess_email(email_text)
print("预处理后的文本：", preprocessed_text)

# 提取自定义关键词
custom_keywords = extract_custom_keywords(email_text, preprocessed_text)
print("自定义关键词：", custom_keywords)

# 示例数据集
data = [
    {'text': 'ok neko meme videos Prof. K June.20th birthday party', 'label': 'social'},
    {'text': 'project deadline June.20th meeting', 'label': 'work'},
    # 添加更多样本
]

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# 关键词向量化
vectorizer, keyword_vectors = vectorize_keywords(texts)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(keyword_vectors, labels, test_size=0.2, random_state=42)

# 训练分类模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)

# 示例新邮件文本
new_email_text = 'Prof. K birthday party on June.20th'
category = classify_email(model, vectorizer, new_email_text)
print("邮件分类结果：", category)

# 发送推送通知
user_id = 'Udc766b4c9cf76ae32b918489c05ba04f'  # 从提供的JSON数据中获取
access_token = '+upv04InwMYYzFrlS58h1dJcoHdX7C8V7+fsboZ/NVSwzVFDNkPb1SSH5m7bMkvgQyZzoaF5Wx8oMrgHbbhJtweAaKseioakTgOjYCN0k/+0Xv0cjZWIrTRlzsJdJ8xMLqEKko1HVwokOlmLNTVGuQdB04t89/1O/w1cDnyilFU='  # 您的频道访问令牌
message = f'You have a new {category} email: {new_email_text}'
status_code = send_line_notification(message, user_id, access_token)
print("推送状态码：", status_code)
