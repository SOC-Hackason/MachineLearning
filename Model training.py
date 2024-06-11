from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# 示例数据集
data = [
    {'text': 'ok neko meme videos Prof. K June.20th birthday party', 'label': 'social'},
    {'text': 'project deadline June.20th meeting', 'label': 'work'},
    # 添加更多样本
]

texts = [item['text'] for item in data]
labels = [item['label'] for item in data]

# 关键词向量化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练分类模型
model = MultinomialNB()
model.fit(X_train, y_train)

# 预测和评估
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("模型准确率：", accuracy)
