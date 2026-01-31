import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 tfidf_dim, num_classes, num_layers=2, dropout=0.3):
       
        super(HybridLanguageModel, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM for sequence processing
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_dim // 2,  
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # TF-IDF processing MLP
        self.tfidf_mlp = nn.Sequential(
            nn.Linear(tfidf_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for BiLSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Combined features processing
        combined_dim = hidden_dim + 256  
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'weight' in name:
                    if 'embedding' in name:
                        nn.init.normal_(param, mean=0, std=0.1)
                    elif 'lstm' in name and param.dim() >= 2:
                        for i in range(0, param.size(0), 4):
                            if i+3 <= param.size(0):
                                nn.init.orthogonal_(param[i:i+3])
                                nn.init.zeros_(param[i+3])
                    elif 'attention' in name or 'classifier' in name or 'tfidf_mlp' in name:
                        if param.dim() >= 2:
                            nn.init.xavier_uniform_(param)
                        elif 'norm' in name:  
                            nn.init.ones_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def apply_attention(self, lstm_output):
        attention_weights = self.attention(lstm_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector
    
    def forward(self, sequences, tfidf_features):
        # Process sequences with BiLSTM
        embedded = self.embedding(sequences)
        embedded = self.dropout(embedded)
        
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        # Apply attention to get context vector
        bilstm_features = self.apply_attention(lstm_out)
        bilstm_features = self.dropout(bilstm_features)
        
        if tfidf_features.dim() == 1:
            tfidf_features = tfidf_features.unsqueeze(0)
        
        tfidf_features = self.tfidf_mlp(tfidf_features)
        tfidf_features = self.dropout(tfidf_features)
        
        # Combine features
        combined = torch.cat([bilstm_features, tfidf_features], dim=1)
        
        # Final classification
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(self, sequences, tfidf_features):
        with torch.no_grad():
            logits = self.forward(sequences, tfidf_features)
            return F.softmax(logits, dim=1)
    
    def predict(self, sequences, tfidf_features):
        probs = self.predict_proba(sequences, tfidf_features)
        return torch.argmax(probs, dim=1)


class EarlyStopping:    
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        return self.early_stop