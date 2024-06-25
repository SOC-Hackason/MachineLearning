import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Read email content
with open('/Users/kiwi/Desktop/mailcontents.txt', 'r', encoding='utf-8') as f:
    email_data = json.load(f)

# Extracts sender, subject, recipient and content
emails = []
for email in email_data:
    combined_text = f"{email['from']} {email['subject']} {email['to']} {email['body']}"
    emails.append(combined_text)

# Create DataFrame
df = pd.DataFrame(emails, columns=['combined_text'])

# Use TF-IDF vetorize text
vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = vectorizer.fit_transform(df['combined_text'])

# Use K-means for clustering
num_clusters = 3  # Adjust the cluster numbers based on need
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_tfidf)

# Get clustering result
df['cluster'] = kmeans.labels_

# Output number of emails for every cluster
print(df['cluster'].value_counts())

# View the content of the first few emails for each cluster
for cluster in range(num_clusters):
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster]['combined_text'].head(10))
