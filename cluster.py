# import sys
# import json
# import os
# import numpy as np
# from sklearn.cluster import DBSCAN
# from sklearn.metrics import pairwise_distances
# from sentence_transformers import SentenceTransformer
# from sklearn.feature_extraction.text import CountVectorizer

# # 1) Read input (list of {path, name} objects)
# files = json.loads(sys.stdin.read())

# # 2) Read file contents
# contents, names = [], []
# for f in files:
#     try:
#         with open(f['path'], 'r', encoding='utf-8') as file:
#             contents.append(file.read())
#     except:
#         contents.append('')
#     names.append(f['name'])

# # 3) Load the model & 4) create embeddings
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(contents)

# # 5) Auto-calc eps
# distances = pairwise_distances(embeddings, metric='cosine')
# np.fill_diagonal(distances, np.inf)
# nearest_distances = np.min(distances, axis=1)
# auto_eps = np.percentile(nearest_distances, 75)

# # 6) DBSCAN
# dbscan = DBSCAN(eps=auto_eps, min_samples=1, metric='cosine')
# labels = dbscan.fit_predict(embeddings)

# # 7) Extract top-10 keywords across all documents
# vectorizer = CountVectorizer(stop_words='english', max_features=10)
# _ = vectorizer.fit_transform(contents)
# keywords = vectorizer.get_feature_names_out()

# # 8) Group names into clusters
# clusters = {}
# for name, lbl in zip(names, labels):
#     clusters.setdefault(str(lbl), []).append(name)

# # 9) Build result, casting NumPy types to native
# result = {
#     "eps": float(auto_eps),           # cast to Python float
#     "min_samples": 1,
#     "keywords": list(keywords),       # already Python strings
#     "clusters": list(clusters.values())
# }

# print(json.dumps(result))

# Top keyword labelling starts here

import sys
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# 1) Read input (list of {path, name} objects)
files = json.loads(sys.stdin.read())

# 2) Read file contents
contents, names = [], []
for f in files:
    try:
        with open(f['path'], 'r', encoding='utf-8') as file:
            contents.append(file.read())
    except:
        contents.append('')
    names.append(f['name'])

# 3) Load the model & create embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(contents)

# 4) Auto-calculate eps
distances = pairwise_distances(embeddings, metric='cosine')
np.fill_diagonal(distances, np.inf)
nearest_distances = np.min(distances, axis=1)
auto_eps = np.percentile(nearest_distances, 75)

# 5) Clustering using DBSCAN
dbscan = DBSCAN(eps=float(auto_eps), min_samples=1, metric='cosine')
labels = dbscan.fit_predict(embeddings)

# 6) Organize documents by cluster
clusters = {}
for content, name, label in zip(contents, names, labels):
    label = str(label)
    if label not in clusters:
        clusters[label] = {
            "files": [],
            "texts": []
        }
    clusters[label]["files"].append(name)
    clusters[label]["texts"].append(content)

# 7) Extract top keywords separately for each cluster
final_clusters = []
for label, data in clusters.items():
    vectorizer = CountVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform(data["texts"])
    keywords = vectorizer.get_feature_names_out()

    final_clusters.append({
        "label": ", ".join(keywords) if keywords.size > 0 else f"Cluster {label}",
        "files": data["files"]
    })

# 8) Build and print final result
result = {
    "eps": float(auto_eps),
    "min_samples": 1,
    "clusters": final_clusters
}

print(json.dumps(result))

