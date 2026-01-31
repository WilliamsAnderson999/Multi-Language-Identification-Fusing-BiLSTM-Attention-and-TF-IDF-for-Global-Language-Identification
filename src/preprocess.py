import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.nn.utils.rnn import pad_sequence
import joblib
import nltk
from nltk.tokenize import word_tokenize
import os

nltk.download('punkt', quiet=True)

class TextPreprocessor:    
    def __init__(self, ngram_range=(1, 3), max_features=5000, max_len=100):
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            analyzer='char',
            lowercase=True
        )
        self.label_encoder = LabelEncoder()
        self.max_len = max_len
        self.vocab = None
        self.word2idx = None
        self.idx2word = None
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u00C0-\u00FF\u0100-\u017F\u0180-\u024F]', '', text)
        return text.strip().lower()
    
    def build_vocab(self, texts):
        all_words = []
        for text in texts:
            tokens = word_tokenize(self.clean_text(text))
            all_words.extend(tokens)
        
        # Get most frequent words
        from collections import Counter
        word_counts = Counter(all_words)
        most_common = word_counts.most_common(20000)  
        
        self.vocab = ['<PAD>', '<UNK>'] + [word for word, _ in most_common]
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
        return self.vocab
    
    def text_to_sequence(self, text):
        if self.word2idx is None:
            raise ValueError("Vocabulary not built. Call build_vocab first.")
        
        tokens = word_tokenize(self.clean_text(text))
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                   for token in tokens[:self.max_len]]
        
        # Padding
        if len(sequence) < self.max_len:
            sequence += [self.word2idx['<PAD>']] * (self.max_len - len(sequence))
        
        return sequence[:self.max_len]
    
    def fit_tfidf(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        self.tfidf.fit(cleaned_texts)
        return self.tfidf.transform(cleaned_texts).toarray()
    
    def transform_tfidf(self, texts):
        cleaned_texts = [self.clean_text(text) for text in texts]
        return self.tfidf.transform(cleaned_texts).toarray()
    
    def fit_labels(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def transform_labels(self, labels):
        return self.label_encoder.transform(labels)
    
    def inverse_transform_labels(self, indices):
        return self.label_encoder.inverse_transform(indices)
    
    def prepare_data(self, texts, labels=None, mode='train'):
        data = {}
        
        if mode == 'train':
            tfidf_features = self.fit_tfidf(texts)
        else:
            tfidf_features = self.transform_tfidf(texts)
        
        data['tfidf'] = tfidf_features  
        
        # Word sequences for BiLSTM
        if mode == 'train':
            self.build_vocab(texts)
        
        sequences = [self.text_to_sequence(text) for text in texts]
        data['sequences'] = np.array(sequences)
        
        # Labels
        if labels is not None:
            if mode == 'train':
                encoded_labels = self.fit_labels(labels)
            else:
                encoded_labels = self.transform_labels(labels)
            data['labels'] = encoded_labels
        
        return data
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'tfidf': self.tfidf,
            'label_encoder': self.label_encoder,
            'vocab': self.vocab,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'max_len': self.max_len
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        preprocessor = cls(max_len=data['max_len'])
        preprocessor.tfidf = data['tfidf']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.vocab = data['vocab']
        preprocessor.word2idx = data['word2idx']
        preprocessor.idx2word = data['idx2word']
        return preprocessor