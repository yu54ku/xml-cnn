import os
import numpy as np
import argparse

def create_bow_embeddings(feature_dim=5000, embedding_dim=300):
    """
    Create special embeddings file that maps feature IDs to embedding vectors.
    These embeddings will be loaded by the XML-CNN model just like GloVe vectors.
    
    For BOW features, we create a mapping where:
    - Each feature ID gets a unique embedding vector
    - The embeddings are initialized with small random values
    """
    print(f"Creating BOW embeddings with {feature_dim} features and {embedding_dim} dimensions...")
    
    os.makedirs(".vector_cache", exist_ok=True)
    
    embeddings = np.random.uniform(-0.25, 0.25, (feature_dim + 2, embedding_dim))
    
    embeddings[0] = 0  
    
    with open(".vector_cache/bow_embeddings.txt", "w") as f:
        f.write(f"pad {' '.join(['0.0'] * embedding_dim)}\n")
        f.write(f"unk {' '.join([str(x) for x in embeddings[1]])}\n")
        
        for i in range(feature_dim):
            token = f"f{i}"
            vector = ' '.join([str(x) for x in embeddings[i + 2]])
            f.write(f"{token} {vector}\n")
    
    print("BOW embeddings created and saved to .vector_cache/bow_embeddings.txt")

def main():
    parser = argparse.ArgumentParser(description="Create embeddings for BOW features")
    parser.add_argument("--dim", type=int, default=5000, help="Feature dimension")
    parser.add_argument("--emb-dim", type=int, default=300, help="Embedding dimension")
    
    args = parser.parse_args()
    create_bow_embeddings(args.dim, args.emb_dim)

if __name__ == "__main__":
    main()