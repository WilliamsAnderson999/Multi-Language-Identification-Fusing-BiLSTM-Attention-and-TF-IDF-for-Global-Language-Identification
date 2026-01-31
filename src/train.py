import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import time
from datetime import datetime
import json
from collections import Counter

from src.preprocess import TextPreprocessor
from src.model import HybridLanguageModel, EarlyStopping

class LanguageIdentifierTrainer:    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.preprocessor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
    def get_default_config(self):
        return {
            # Model hyperparameters
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            
            # Training hyperparameters
            'batch_size': 64,
            'learning_rate': 0.0005,
            'epochs': 40,
            'weight_decay': 1e-5,
            
            # TF-IDF settings
            'ngram_range': (1, 3),
            'max_tfidf_features': 3000,
            'max_seq_len': 100,
            
            # Early stopping
            'patience': 15,
            'min_delta': 0.001,
            
            # Data settings
            'val_split': 0.1,
            'random_seed': 42
        }
    
    def load_data(self, data_dir='data'):        
        with open(os.path.join(data_dir, 'x_train.txt'), 'r', encoding='utf-8') as f:
            X_train = [line.strip() for line in f]
        
        with open(os.path.join(data_dir, 'y_train.txt'), 'r', encoding='utf-8') as f:
            y_train = [line.strip() for line in f]
        
        with open(os.path.join(data_dir, 'x_test.txt'), 'r', encoding='utf-8') as f:
            X_test = [line.strip() for line in f]
        
        with open(os.path.join(data_dir, 'y_test.txt'), 'r', encoding='utf-8') as f:
            y_test = [line.strip() for line in f]
        
        with open(os.path.join(data_dir, 'labels.csv'), 'r', encoding='utf-8') as f:
            labels = [line.strip() for line in f]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of languages: {len(labels)}")
        
        return X_train, y_train, X_test, y_test, labels
    
    def prepare_data(self, X_train, y_train, X_test, y_test):
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(
            ngram_range=self.config['ngram_range'],
            max_features=self.config['max_tfidf_features'],
            max_len=self.config['max_seq_len']
        )
        
        train_data = self.preprocessor.prepare_data(X_train, y_train, mode='train')
        test_data = self.preprocessor.prepare_data(X_test, y_test, mode='test')
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(X_train))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=self.config['val_split'],
            random_state=self.config['random_seed'],
            stratify=train_data['labels']
        )
        
        # Create data splits
        data_splits = {}
        for split_name, idx in [('train', train_idx), ('val', val_idx)]:
            data_splits[split_name] = {
                'sequences': train_data['sequences'][idx],
                'tfidf': train_data['tfidf'][idx],
                'labels': train_data['labels'][idx]
            }
        
        # Test data
        data_splits['test'] = {
            'sequences': test_data['sequences'],
            'tfidf': test_data['tfidf'],
            'labels': test_data['labels']
        }
        
        print(f"Train samples: {len(data_splits['train']['labels'])}")
        print(f"Validation samples: {len(data_splits['val']['labels'])}")
        print(f"Test samples: {len(data_splits['test']['labels'])}")
        
        return data_splits
    
    def create_dataloaders(self, data_splits):
        dataloaders = {}
        
        for split in ['train', 'val', 'test']:
            data = data_splits[split]
            
            # Convert to tensors
            sequences_tensor = torch.LongTensor(data['sequences'])
            tfidf_tensor = torch.FloatTensor(data['tfidf'])
            labels_tensor = torch.LongTensor(data['labels'])
            
            # Create dataset
            dataset = TensorDataset(sequences_tensor, tfidf_tensor, labels_tensor)
            
            # Create dataloader
            batch_size = self.config['batch_size']
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=2,
                pin_memory=True if self.device.type == 'cuda' else False
            )
        
        return dataloaders
    
    def build_model(self, vocab_size, num_classes):        
        self.model = HybridLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            tfidf_dim=self.config['max_tfidf_features'],
            num_classes=num_classes,
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        self.model = self.model.to(self.device)
        
        # Model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
        for batch_idx, (sequences, tfidf, labels) in enumerate(progress_bar):
            sequences = sequences.to(self.device, non_blocking=True)
            tfidf = tfidf.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            outputs = self.model(sequences, tfidf)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if batch_idx % 50 == 0:
                current_acc = accuracy_score(preds.cpu().numpy(), labels.cpu().numpy())
                if self.device.type == 'cuda':
                    memory_allocated = torch.cuda.memory_allocated() / 1e9
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.4f}',
                        'GPU_mem': f'{memory_allocated:.2f}GB'
                    })
                else:
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.4f}'
                    })
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, tfidf, labels in dataloader:
                sequences = sequences.to(self.device)
                tfidf = tfidf.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(sequences, tfidf)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                probs = torch.softmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return avg_loss, accuracy, f1, all_preds, all_labels, all_probs
    
    def train(self, dataloaders, save_dir='model'):        
        # Initialize model
        vocab_size = len(self.preprocessor.vocab)
        num_classes = len(self.preprocessor.label_encoder.classes_)
        self.build_model(vocab_size, num_classes)
        
        # optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        early_stopping = EarlyStopping(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )
        
        best_val_acc = 0
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc, train_f1 = self.train_epoch(
                dataloaders['train'], epoch
            )
            
            val_loss, val_acc, val_f1, _, _, _ = self.evaluate(dataloaders['val'])
            
            self.scheduler.step(val_acc)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(os.path.join(save_dir, 'best_model.pth'))
                print(f"  ✓ New best model saved (val_acc: {val_acc:.4f})")
            
            # Check early stopping
            if early_stopping(-val_acc):  
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self.load_model(os.path.join(save_dir, 'best_model.pth'))
        
        # Final evaluation
        test_loss, test_acc, test_f1, test_preds, test_labels, test_probs = self.evaluate(
            dataloaders['test']
        )
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        
        return test_preds, test_labels, test_probs
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        
        preprocessor_path = os.path.join(os.path.dirname(path), 'preprocessor.pkl')
        self.preprocessor.save(preprocessor_path)
        
        config_path = os.path.join(os.path.dirname(path), 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"  Model saved to: {path}")
        print(f"  Preprocessor saved to: {preprocessor_path}")
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        
        vocab_size = len(self.preprocessor.vocab)
        num_classes = len(self.preprocessor.label_encoder.classes_)
        self.build_model(vocab_size, num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.config = checkpoint.get('config', self.config)
    
    def plot_training_history(self, save_path=None):
        if len(self.history['train_loss']) == 0:
            print("No training history to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(self.history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, max_languages=20):
        if len(y_true) == 0 or len(y_pred) == 0:
            print("No data for confusion matrix")
            return
        
        languages = self.preprocessor.label_encoder.classes_
        
        # Find most frequent languages in predictions
        pred_counts = Counter(y_pred)
        top_languages_idx = [idx for idx, _ in pred_counts.most_common(max_languages)]
        top_languages = [languages[idx] for idx in top_languages_idx]
        
        # Filter data for top languages
        mask = np.isin(y_true, top_languages_idx)
        filtered_y_true = y_true[mask]
        filtered_y_pred = y_pred[mask]
        
        # Create confusion matrix
        cm = confusion_matrix(filtered_y_true, filtered_y_pred)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=top_languages,
                   yticklabels=top_languages)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Top {len(top_languages)} Languages')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def generate_detailed_report(self, test_preds, test_labels, save_dir='model'):
        target_names = self.preprocessor.label_encoder.classes_
        num_languages = len(target_names)
        
        # Generate classification report
        print(f"\nClassification Report for {num_languages} languages:")
        
        if num_languages > 30:
            label_counts = Counter(test_labels)
            top_30_idx = [idx for idx, _ in label_counts.most_common(30)]
            top_30_names = [target_names[idx] for idx in top_30_idx]
            
            mask = np.isin(test_labels, top_30_idx)
            filtered_preds = test_preds[mask]
            filtered_labels = test_labels[mask]
            
            print(classification_report(filtered_labels, filtered_preds, 
                                        target_names=top_30_names, digits=4))
        else:
            print(classification_report(test_labels, test_preds, 
                                        target_names=target_names, digits=4))
        
        # Calculate per-language accuracy
        print(f"\n Per-language accuracy (top 20):")
        language_accuracies = {}
        
        for lang_idx in range(num_languages):
            mask = test_labels == lang_idx
            if mask.sum() > 0:
                lang_acc = accuracy_score(test_labels[mask], test_preds[mask])
                language_accuracies[target_names[lang_idx]] = lang_acc
        
        sorted_accuracies = sorted(language_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        for i, (lang_name, acc) in enumerate(sorted_accuracies[:20]):
            print(f"  {i+1:2d}. {lang_name[:30]:30} {acc:.4f}")
        
        if len(sorted_accuracies) > 20:
            print(f"  ... and {len(sorted_accuracies) - 20} more languages")
        
        results = {
            'test_accuracy': accuracy_score(test_labels, test_preds),
            'test_f1': f1_score(test_labels, test_preds, average='weighted'),
            'num_languages': num_languages,
            'vocab_size': len(self.preprocessor.vocab),
            'per_language_accuracy': {k: float(v) for k, v in language_accuracies.items()},
            'config': self.config,
            'training_time': self.history.get('training_time', 0),
            'epochs_trained': len(self.history['train_loss'])
        }
        
        results_path = os.path.join(save_dir, 'detailed_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n Detailed results saved to: {results_path}")
        
        return results


def main():
    print("=" * 60)
    print("LANGUAGE IDENTIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Initialize trainer
    trainer = LanguageIdentifierTrainer()
    
    # Load data
    X_train, y_train, X_test, y_test, labels = trainer.load_data('data')
    
    # Prepare data
    data_splits = trainer.prepare_data(X_train, y_train, X_test, y_test)
    
    # Create dataloaders
    dataloaders = trainer.create_dataloaders(data_splits)
    
    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    test_preds, test_labels, test_probs = trainer.train(
        dataloaders, 
        save_dir='model'
    )
    
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS AND REPORTS")
    print("=" * 60)
    
    trainer.plot_training_history('model/training_history.png')
    
    trainer.plot_confusion_matrix(test_labels, test_preds, 
                                  'model/confusion_matrix_top20.png',
                                  max_languages=20)
    
    results = trainer.generate_detailed_report(test_preds, test_labels, 'model')
    
    summary = {
        'model_name': 'Hybrid TF-IDF + BiLSTM Language Identifier',
        'dataset': 'WiLI-2018',
        'test_accuracy': results['test_accuracy'],
        'test_f1_score': results['test_f1'],
        'num_languages': results['num_languages'],
        'best_languages': list(results['per_language_accuracy'].items())[:10],
        'worst_languages': list(sorted(results['per_language_accuracy'].items(), 
                                       key=lambda x: x[1]))[:10],
        'training_info': {
            'epochs': results['epochs_trained'],
            'batch_size': trainer.config['batch_size'],
            'learning_rate': trainer.config['learning_rate']
        }
    }
    
    summary_path = 'model/training_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("LANGUAGE IDENTIFICATION TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {summary['model_name']}\n")
        f.write(f"Dataset: {summary['dataset']}\n")
        f.write(f"Number of languages: {summary['num_languages']}\n\n")
        f.write(f"FINAL RESULTS:\n")
        f.write(f"  Test Accuracy: {summary['test_accuracy']:.4%}\n")
        f.write(f"  Test F1 Score: {summary['test_f1_score']:.4%}\n\n")
        f.write(f" BEST PERFORMING LANGUAGES (Top 10):\n")
        for i, (lang, acc) in enumerate(summary['best_languages'], 1):
            f.write(f"  {i:2d}. {lang[:30]:30} {acc:.4%}\n")
        f.write(f"\n MOST CHALLENGING LANGUAGES (Bottom 10):\n")
        for i, (lang, acc) in enumerate(summary['worst_languages'], 1):
            f.write(f"  {i:2d}. {lang[:30]:30} {acc:.4%}\n")
        f.write(f"\n TRAINING CONFIGURATION:\n")
        f.write(f"  Epochs trained: {summary['training_info']['epochs']}\n")
        f.write(f"  Batch size: {summary['training_info']['batch_size']}\n")
        f.write(f"  Learning rate: {summary['training_info']['learning_rate']}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"\n Training summary saved to: {summary_path}")
    
    print("\n" + "=" * 60)
    print("TESTING WITH EXAMPLE SENTENCES")
    print("=" * 60)
    
    example_sentences = [
        ("Bonjour, comment allez-vous aujourd'hui?", "French"),
        ("Hello, how are you doing today?", "English"),
        ("Hola, ¿cómo estás hoy?", "Spanish"),
        ("Guten Tag, wie geht es Ihnen heute?", "German"),
        ("Ciao, come stai oggi?", "Italian"),
        ("Здравствуйте, как у вас дела сегодня?", "Russian"),
        ("今日は、元気ですか？", "Japanese"),
        ("你好，今天过得怎么样？", "Chinese")
    ]
    
    for text, expected_lang in example_sentences:
        data = trainer.preprocessor.prepare_data([text], mode='test')
        
        sequences = torch.LongTensor(data['sequences'])
        tfidf = torch.FloatTensor(data['tfidf'])
        
        with torch.no_grad():
            trainer.model.eval()
            outputs = trainer.model(sequences, tfidf)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item() * 100
        
        predicted_lang = trainer.preprocessor.inverse_transform_labels([pred_idx])[0]
        
        status = "✓" if predicted_lang.lower() == expected_lang.lower() else "✗"
        print(f"  {status} '{text[:40]}...'")
        print(f"     → Predicted: {predicted_lang} ({confidence:.1f}%)")
        print(f"     → Expected: {expected_lang}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)

if __name__ == "__main__":
    main()