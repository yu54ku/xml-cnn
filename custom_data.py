import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class XMLCNNDataset(Dataset):
    def __init__(self, file_path, vocab=None, build_vocab=True, max_length=500, label_list=None, label_to_idx=None):
        self.ids = []
        self.texts = []
        self.labels = []
        self.max_length = max_length
        self.label_set = set()
        
        print(f"Loading data from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    doc_id, label_str, text = parts
                    self.ids.append(doc_id)
                    self.labels.append(label_str.split())
                    self.texts.append(text.split())
                    self.label_set.update(label_str.split())
        
        if label_list is not None:
            self.label_list = label_list
            self.label_to_idx = label_to_idx
        else:
            self.label_list = sorted(list(self.label_set))
            self.label_to_idx = {label: i for i, label in enumerate(self.label_list)}
        
        if build_vocab:
            if vocab is None:
                self.vocab = {'<pad>': 0, '<unk>': 1}
                for doc in self.texts:
                    for token in doc:
                        if token not in self.vocab:
                            self.vocab[token] = len(self.vocab)
            else:
                self.vocab = vocab
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.vocab.get(token, self.vocab['<unk>']) for token in text]
        
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<pad>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        label_indices = []
        for label in self.labels[idx]:
            if label in self.label_to_idx:  
                label_indices.append(self.label_to_idx[label])
        
        label_vector = np.zeros(len(self.label_list))
        label_vector[label_indices] = 1
        
        return {
            'id': self.ids[idx],
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label_vector, dtype=torch.float)
        }
    
    def load_vectors(self, vector_file, dim=300):
        """Load pre-trained word vectors - supports both GloVe and BOW embeddings"""
        print(f"Loading word vectors from {vector_file}...")
        
        is_bow = 'bow_embeddings.txt' in vector_file
        
        vectors = {}
        with open(vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                vectors[word] = vector
        
        embedding_matrix = np.random.uniform(-0.25, 0.25, (len(self.vocab), dim))
        embedding_matrix[0] = 0  
        
        for word, idx in self.vocab.items():
            if word in vectors:
                embedding_matrix[idx] = vectors[word]
            elif is_bow and word.startswith('f') and word[1:].isdigit():
                feature_id = int(word[1:])
                feature_token = f"f{feature_id}"
                if feature_token in vectors:
                    embedding_matrix[idx] = vectors[feature_token]
        
        return torch.FloatTensor(embedding_matrix)

class Batch:
    def __init__(self, batch_data, device=None):
        self.id = [item['id'] for item in batch_data]
        self.text = torch.stack([item['text'] for item in batch_data])
        self.label = torch.stack([item['label'] for item in batch_data])
        
        if device:
            self.text = self.text.to(device)
            self.label = self.label.to(device)
    
    def to(self, device):
        self.text = self.text.to(device)
        self.label = self.label.to(device)
        return self

def collate_batch(batch):
    return Batch(batch)

class Iterator:
    def __init__(self, dataset, batch_size, device=None, train=True, sort=False):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            collate_fn=collate_batch
        )
        self.device = device
    
    def __iter__(self):
        for batch in self.dataloader:
            if self.device:
                batch.to(self.device)
            yield batch