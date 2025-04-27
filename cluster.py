import sys
import json
import re
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import mammoth
import pdfplumber

# 0) Setup
model = SentenceTransformer('all-mpnet-base-v2')
junk_phrases = ["abstract","introduction","conclusion","references",
                "table of contents","acknowledgments","index","appendix"]
def clean_text(text):
    t = text.lower()
    t = re.sub(r'\d+','',t)
    t = re.sub(r'[^\w\s]','',t)
    t = re.sub(r'\s+',' ',t)
    for p in junk_phrases:
        t = t.replace(p,'')
    return t.strip()

# 1) PDF / DOCX / TXT extraction
def extract_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)
def extract_docx(path):
    with open(path,'rb') as f:
        return mammoth.extract_raw_text(f).value
def extract_all(entry):
    p, name = entry['path'], entry['name']
    if p.lower().endswith('.pdf'):
        return name, extract_pdf(p)
    if p.lower().endswith('.docx'):
        return name, extract_docx(p)
    # fallback to txt
    return name, open(p, encoding='utf-8', errors='ignore').read()

# 2) Read + extract + clean
files = json.loads(sys.stdin.read())
names, texts = zip(*[  
    (n, clean_text(txt))
    for n, txt in map(extract_all, files)
])

# 3) Embed **all** cleaned texts
embs = model.encode(texts, show_progress_bar=False,
                    batch_size=16, convert_to_numpy=True)

# 4) Deduplicate by embedding similarity
unique_names = []
unique_embs  = []
for name, emb in zip(names, embs):
    if not unique_embs:
        unique_names.append(name)
        unique_embs.append(emb)
        continue
    sims = cosine_similarity([emb], unique_embs)[0]
    if sims.max() < 0.95:
        unique_names.append(name)
        unique_embs.append(emb)
# convert back to numpy
unique_embs = np.vstack(unique_embs)

# 5) Compute eps for DBSCAN
dists = pairwise_distances(unique_embs, metric='cosine')
np.fill_diagonal(dists, np.inf)
eps = float(np.percentile(np.min(dists, axis=1), 75))

# 6) DBSCAN clustering
labels = DBSCAN(eps=eps, min_samples=1, metric='cosine').fit_predict(unique_embs)

# 7) Group names by cluster
clusters = {}
for lbl, nm in zip(labels, unique_names):
    clusters.setdefault(str(lbl), []).append(nm)

# 8) Extract 3 TF-IDF keywords per cluster
final = []
for lbl, group in clusters.items():
    # gather original texts for this group
    grp_texts = [texts[names.index(n)] for n in group]
    vec = TfidfVectorizer(stop_words='english', max_features=3)
    X = vec.fit_transform(grp_texts)
    kws = vec.get_feature_names_out()
    label = ", ".join(kws) if len(kws) else f"Cluster {lbl}"
    final.append({"label": label, "files": group})

# 9) Output
out = {"eps": eps, "min_samples": 1, "clusters": final}
sys.stdout.write(json.dumps(out))
