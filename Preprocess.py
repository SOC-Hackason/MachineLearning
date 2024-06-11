import re
import spacy
from collections import Counter

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
    email_text = re.sub(r'(prof.|sommelier|dear|regards|best regards|sincerely|hello|hi|thank you|thanks|yours|truly|faithfully|cheers|kind regards)[\s,]*', '', email_text)

    # 分词与词性标注
    doc = nlp(email_text)
    # 词形还原并去除停用词
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def extract_custom_keywords(email_text):
    # 使用spacy进行命名实体识别
    doc = nlp(email_text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 提取特定关键词
    keywords = []
    for ent in entities:
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'DATE']:
            keywords.append(ent[0])
    
    # 查找特定短语
    phrases = ['neko meme videos', 'birthday party', 'ok']
    for phrase in phrases:
        if phrase in email_text.lower():
            keywords.append(phrase)

    # 统计高频词
    word_freq = Counter(email_text.split())
    high_freq_words = [word for word, freq in word_freq.items() if freq > 1 and word not in phrases]

    keywords.extend(high_freq_words)
    
    # 去重
    keywords = list(set(keywords))
    return keywords

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
custom_keywords = extract_custom_keywords(email_text)
print("自定义关键词：", custom_keywords)
