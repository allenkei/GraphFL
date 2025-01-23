import gensim.downloader as api
import numpy as np

model = api.load("glove-wiki-gigaword-100")
data = np.load('data/adjnoun_raw.npz', allow_pickle=True) # from R

adj_matrix = data['adj'] # the adjacency matrix
y_data = data['labels'] # the actual word
y_classes = data['classes'] # 0 is adj, 1 is noun

adjnoun_vectors = [model[word] if word in model else np.zeros(100) for word in y_data]
adjnoun_vectors = np.array(adjnoun_vectors) # word embeddings

np.savez('data/adjnoun_vec.npz', adj=adj_matrix, labels=y_data, classes=y_classes, word_vec = adjnoun_vectors)

