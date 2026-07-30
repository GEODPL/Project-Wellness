from sentence_transformers import SentenceTransformer

print("Φορτώνω το μοντέλο (πρώτη φορά θα κατέβει, ~470MB)...")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

sentences = [
    "Αποτυγχάνω σε όλα.",
    "Δεν τα καταφέρνω ποτέ.",
    "Σήμερα έφαγα ένα νόστιμο σουβλάκι.",
]

embeddings = model.encode(sentences)

print(f"\nΔιαστάσεις κάθε embedding: {embeddings.shape}")
print(f"Παράδειγμα (πρώτα 5 νούμερα της 1ης πρότασης): {embeddings[0][:5]}")

# Μοιάζουν σημασιολογικά οι δύο πρώτες προτάσεις;
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

print(f"\nΟμοιότητα 'Αποτυγχάνω σε όλα' <-> 'Δεν τα καταφέρνω ποτέ': {sim_1_2:.3f}")
print(f"Ομοιότητα 'Αποτυγχάνω σε όλα' <-> 'Έφαγα σουβλάκι':        {sim_1_3:.3f}")