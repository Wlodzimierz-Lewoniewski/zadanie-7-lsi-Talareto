import numpy as np
import re

def clean_text(text):
    # Remove punctuation and extra spaces, and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip().lower()

def lsi_similarity(n, documents, query, k):
    # Tokenization
    terms = set(word for doc in documents + [query] for word in doc.split())
    terms = sorted(terms)  # Maintain consistent ordering
    term_index = {term: i for i, term in enumerate(terms)}

    # Term-document matrix
    C = np.zeros((len(terms), n), dtype=int)
    for j, doc in enumerate(documents):
        for word in doc.split():
            if word in term_index:
                C[term_index[word], j] = 1

    # Singular Value Decomposition
    U, s, Vt = np.linalg.svd(C, full_matrices=False)

    # Reduce dimensions to k
    sr = np.copy(s)
    sr[k:] = 0  # Set the smallest singular values to 0
    Sr = np.diag(sr)

    # Reduced term-document matrix
    Cr = U.dot(Sr).dot(Vt)

    # Transform query vector into reduced space
    q_vec = np.zeros(len(terms), dtype=int)
    for word in query.split():
        if word in term_index:
            q_vec[term_index[word]] = 1

    # Transform query using reduced dimensions
    Sk = np.diag(s[:k])
    UkT = np.take(U.T, range(k), axis=0)
    qk = np.linalg.inv(Sk).dot(UkT).dot(q_vec)

    # Compute cosine similarity between query and documents
    Vk = np.take(Vt, range(k), axis=0)
    doc_vectors = Sk.dot(Vk).T

    similarities = []
    for doc_vec in doc_vectors:
        cosine_sim = np.dot(qk, doc_vec) / (np.linalg.norm(qk) * np.linalg.norm(doc_vec))
        similarities.append(round(cosine_sim, 2))

    return similarities

if __name__ == "__main__":
    # User input
    n = int(input("Enter the number of documents: "))
    print("Enter each document on a new line:")
    documents = [clean_text(input(f"Document {i+1}: ")) for i in range(n)]

    query = clean_text(input("Enter the query: "))
    k = int(input("Enter the number of dimensions to reduce to (k): "))

    # Calculate similarities
    similarities = lsi_similarity(n, documents, query, k)
    print("\nSimilarities:")
    for i, sim in enumerate(similarities):
        print(f"Document {i+1}: {sim}")
