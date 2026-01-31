# Language Identification System  
**Hybrid TF-IDF + BiLSTM model** – 235 languages – 93.75% accuracy

<p align="center">
  <img src="https://github.com/yourusername/language-identification/raw/main/images/Model%20Architecture.png" width="720"><br>
  <i>Figure 1: Hybrid TF-IDF + BiLSTM Architecture with Attention Mechanism</i>
</p>

State-of-the-art multilingual language identification system trained on the **WiLI-2018** benchmark (235 languages).

## Highlights

- **235 languages** supported  
- Hybrid model: **TF-IDF character n-grams** + **BiLSTM + Attention**  
- **93.75%** overall test accuracy (117,500 samples)  
- Real-time web interface with **Gradio**  
- File support: **.txt**, **.pdf**, **.docx**  
- Confidence visualization & detailed performance reports

## Model Performance

| Metric              | Value          |
|---------------------|----------------|
| Test Accuracy       | 93.75%         |
| Test Macro F1       | 93.75%         |
| Vocabulary Size     | 20,002         |
| Total Parameters    | ~4.2 M         |
| Inference Time      | < 100 ms       |
| Web Response Time   | < 500 ms       |

### Performance Distribution

| Category       | Accuracy Range | # Languages | Percentage |
|----------------|----------------|-------------|------------|
| Excellent      | ≥ 99%          | 25          | 10.6%      |
| Good           | 95–99%         | 131         | 55.7%      |
| Average        | 80–95%         | 68          | 28.9%      |
| Challenging    | < 80%          | 11          | 4.7%       |

### Top 10 Performing Languages

| Rank | Code | Language            | Accuracy | Samples |
|------|------|---------------------|----------|---------|
| 1    | ckb  | Sorani Kurdish      | 100%     | 500     |
| 2    | kbd  | Kabardian           | 100%     | 500     |
| 3    | min  | Minangkabau         | 100%     | 500     |
| 4    | mlg  | Malagasy            | 100%     | 500     |
| 5    | bod  | Tibetan             | 99.8%    | 500     |
| 6    | ceb  | Cebuano             | 99.8%    | 500     |
| 7    | div  | Dhivehi             | 99.8%    | 500     |
| 8    | jbo  | Lojban              | 99.8%    | 500     |
| 9    | mri  | Maori               | 99.8%    | 500     |
| 10   | nav  | Navajo              | 99.8%    | 500     |

### Most Challenging Languages

| Rank | Code   | Language          | Accuracy | Family   |
|------|--------|-------------------|----------|----------|
| 1    | wuu    | Wu Chinese        | 15.6%    | Asian    |
| 2    | zh-yue | Cantonese         | 22.8%    | Asian    |
| 3    | zho    | Mandarin          | 37.0%    | Asian    |
| 4    | hrv    | Croatian          | 46.8%    | Slavic   |
| 5    | hbs    | Serbo-Croatian    | 53.8%    | Slavic   |

## Quick Start

```bash
git clone https://github.com/yourusername/language-identification.git
cd language-identification

# 1. Virtual environment
python -m venv venv
source venv/bin/activate    # Linux / macOS
# or
venv\Scripts\activate       # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (only once)
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"