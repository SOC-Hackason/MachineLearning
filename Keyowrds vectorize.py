from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_keywords(keywords):
    vectorizer = TfidfVectorizer()
    keyword_vectors = vectorizer.fit_transform(keywords)
    return vectorizer, keyword_vectors

# 示例关键词列表
keywords = [
    'ok', 'neko meme videos', 'Prof. K', 'June.20th', 'birthday party'
]

vectorizer, keyword_vectors = vectorize_keywords(keywords)
print("关键词向量：", keyword_vectors.toarray())
