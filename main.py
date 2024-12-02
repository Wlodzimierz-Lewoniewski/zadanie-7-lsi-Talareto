import numpy as np
import re

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

def lsi_analysis(num_docs, doc_list, search_query, dimensions):
    vocab = set(word for doc in doc_list + [search_query] for word in doc.split())
    vocab = sorted(vocab)
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    term_doc_matrix = np.zeros((len(vocab), num_docs), dtype=int)
    for idx, doc in enumerate(doc_list):
        for word in doc.split():
            if word in word_to_index:
                term_doc_matrix[word_to_index[word], idx] = 1

    U, s, Vt = np.linalg.svd(term_doc_matrix, full_matrices=False)

    truncated_singular_values = np.copy(s)
    truncated_singular_values[dimensions:] = 0
    reduced_S = np.diag(truncated_singular_values)

    reduced_matrix = U @ reduced_S @ Vt

    query_vector = np.zeros(len(vocab), dtype=int)
    for word in search_query.split():
        if word in word_to_index:
            query_vector[word_to_index[word]] = 1

    reduced_query_vector = np.linalg.inv(reduced_S).dot(U.T[:dimensions]).dot(query_vector)

    reduced_Vt = Vt[:dimensions]
    doc_vectors = reduced_S @ reduced_Vt.T

    similarity_scores = []
    for vector in doc_vectors.T:
        similarity = np.dot(reduced_query_vector, vector) / (np.linalg.norm(reduced_query_vector) * np.linalg.norm(vector))
        similarity_scores.append(round(similarity, 2))

    return similarity_scores

if __name__ == "__main__":
    num_docs = int(input("Enter the number of documents: "))
    print("Enter each document on a new line:")
    documents = [preprocess_text(input(f"Document {i+1}: ")) for i in range(num_docs)]

    query_text = preprocess_text(input("Enter the query: "))
    dimension_count = int(input("Enter the number of dimensions to reduce to (k): "))

    results = lsi_analysis(num_docs, documents, query_text, dimension_count)
    print("\nDocument Similarities:")
    for i, score in enumerate(results):
        print(f"Document {i+1}: {score}")
