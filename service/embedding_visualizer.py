import gensim.downloader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the word embedding model
model = gensim.downloader.load("glove-wiki-gigaword-50")

# Select words to visualize
words = ["cat", "dog", "tiger", "lion", "boy", "rabbit", "queen", "fish", "king", "girl"]

# Get vectors for selected words
word_vectors = np.array([model[word] for word in words])

# Reduce dimensions to 2D using PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(word_vectors)

# Plot the words in 2D space
plt.figure(figsize=(8, 6))
for word, (x, y) in zip(words, reduced_vectors):
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, word, fontsize=12)

plt.title("2D Visualization of Word Embeddings")
plt.grid()
plt.show()
