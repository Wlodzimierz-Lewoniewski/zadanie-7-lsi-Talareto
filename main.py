import numpy as np
import re

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def lsi_similarity(n, documents, query, k):
    terms = set(word for doc in documents + [query] for word in doc.split())
    terms = sorted(terms)
    term_index = {term: i for i, term in enumerate(terms)}

    C = np.zeros((len(terms), n), dtype=int)
    for j, doc in enumerate(documents):
        for word in doc.split():
            if word in term_index:
                C[term_index[word], j] = 1

    U, s, Vt = np.linalg.svd(C, full_matrices=False)

    sr = np.copy(s)
    sr[k:] = 0
    Sr = np.diag(sr)

    Cr = U.dot(Sr).dot(Vt)

    q_vec = np.zeros(len(terms), dtype=int)
    for word in query.split():
        if word in term_index:
            q_vec[term_index[word]] = 1

    Sk = np.diag(s[:k])
    UkT = np.take(U.T, range(k), axis=0)
    qk = np.linalg.pinv(Sk).dot(UkT).dot(q_vec)

    Vk = np.take(Vt, range(k), axis=0)
    doc_vectors = Sk.dot(Vk).T

    similarities = []
    for doc_vec in doc_vectors:
        cosine_sim = np.dot(qk, doc_vec) / (np.linalg.norm(qk) * np.linalg.norm(doc_vec))
        similarities.append(float(cosine_sim))  # Convert to float for compatibility

    return similarities

if __name__ == "__main__":
    n = int(input().strip())
    documents = [clean_text(input().strip()) for _ in range(n)]
    query = clean_text(input().strip())
    k = int(input().strip())

    similarities = lsi_similarity(n, documents, query, k)
    print(similarities)
