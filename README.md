# 🤗 Empathetic Chatbot - Transformer from Scratch

A complete implementation of a Transformer encoder-decoder model built from scratch for generating empathetic responses. This project demonstrates deep learning fundamentals without using pre-trained models.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## 🎯 Overview

This project implements a **Transformer-based empathetic chatbot** trained on the Facebook AI Empathetic Dialogues dataset. The model learns to generate contextually appropriate and emotionally intelligent responses based on:
- **Emotion context** (32 emotions like sentimental, joyful, afraid, etc.)
- **Situation description** (background context)
- **User message** (customer utterance)

### Key Highlights

✅ **Built from scratch** - No pre-trained models (GPT, BERT, etc.)  
✅ **Full Transformer architecture** - Encoder-decoder with multi-head attention  
✅ **Teacher forcing training** - Standard seq2seq training approach  
✅ **Multiple decoding strategies** - Greedy & Beam search  
✅ **Comprehensive evaluation** - BLEU, ROUGE-L, chrF, Perplexity  
✅ **Interactive web app** - Streamlit UI with conversation history  
✅ **Google Drive integration** - Automatic model download

---

## 🏗️ Architecture

### Transformer Encoder-Decoder

The model implements the classic Transformer architecture from "Attention is All You Need" (Vaswani et al., 2017):

```
Input Text → Encoder → Encoder Output → Decoder → Output Text
                ↓                           ↑
          Positional Encoding    Cross-Attention
                ↓                           ↑
          Multi-Head Attention    Self-Attention (Causal)
                ↓                           ↑
          Feed-Forward Network    Feed-Forward Network
```

### Components Implemented

1. **Positional Encoding** - Sinusoidal position embeddings
2. **Multi-Head Attention** - Scaled dot-product attention with multiple heads
3. **Feed-Forward Network** - Position-wise fully connected layers
4. **Layer Normalization** - Normalization for stable training
5. **Residual Connections** - Skip connections for gradient flow
6. **Causal Masking** - Prevents decoder from seeing future tokens

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 256 |
| Attention Heads | 4 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Feed-Forward Dimension | 512 |
| Dropout | 0.1 |
| Max Sequence Length | 128 |
| Vocabulary Size | ~10,000 tokens |

---

## ✨ Features

### Core Features

- **Emotion-Aware Responses** - Understands 32 different emotional contexts
- **Context Understanding** - Processes situation descriptions and user messages
- **Teacher Forcing Training** - Standard seq2seq training methodology
- **Multiple Decoding Strategies**:
  - Greedy decoding (fast, deterministic)
  - Beam search (better quality, configurable beam width 2-5)

### Web Application Features

- **Interactive Chat Interface** - User-friendly Streamlit UI
- **Emotion Selection** - Choose from 32 emotions
- **Conversation History** - Track multiple exchanges
- **Response Quality Settings** - Select decoding strategy
- **Example Inputs** - Pre-filled examples for quick testing
- **Auto-Download Models** - Downloads from Google Drive if not present

---

## 📊 Dataset

**Empathetic Dialogues** by Facebook AI Research

- **Source**: [Facebook AI Empathetic Dialogues](https://github.com/facebookresearch/EmpatheticDialogues)
- **Size**: 25,000+ conversations
- **Emotions**: 32 emotion categories
- **Structure**: Situation → Customer utterance → Agent response

### Data Format

```
Emotion: sentimental
Situation: I remember going to the fireworks with my best friend
Customer: This was a best friend. I miss her.
Agent: Where has she gone?
```

### Data Split

- **Training**: 80% (used for vocabulary building and training)
- **Validation**: 10% (hyperparameter tuning and checkpoint selection)
- **Test**: 10% (final evaluation)

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd empathetic-chatbot
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt** contains:
```
streamlit==1.28.0
torch==2.0.1
numpy==1.24.3
pandas==2.0.3
gdown==4.7.1
nltk
rouge-score
scikit-learn
```

### Step 3: Download Model Files

#### Option A: Automatic Download (Recommended)
The app automatically downloads models from Google Drive on first run.

#### Option B: Manual Download
1. Download `best_transformer_model.pt` and `vocabulary.pkl`
2. Place them in the project directory

---

## 💻 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Chatbot

1. **Select Emotion** - Choose from 32 emotions in the sidebar
2. **Enter Situation** - Describe the context/background
3. **Enter Message** - Type your message
4. **Choose Decoding** - Select Greedy or Beam Search
5. **Generate Response** - Click the button to get AI response

### Training from Scratch

Open and run the Jupyter notebook:

```bash
jupyter notebook empathetic_chatbot.ipynb
```

Or on **Kaggle**:
1. Upload the notebook to Kaggle
2. Add the Empathetic Dialogues dataset
3. Enable GPU accelerator
4. Run all cells

---

## 🔧 Model Details

### Input Format

```
"Emotion: {emotion} | Situation: {situation} | Customer: {message} Agent:"
```

### Output Format

```
"{empathetic_response}"
```

### Preprocessing

1. **Text Normalization**:
   - Convert to lowercase
   - Normalize whitespace
   - Keep basic punctuation
   
2. **Tokenization**:
   - Simple regex-based tokenization
   - Splits on whitespace and punctuation
   
3. **Vocabulary**:
   - Built from training data only
   - Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`
   - Minimum frequency: 2
   - Max size: 10,000 tokens

### Special Tokens

- `<pad>` (0) - Padding token for variable length sequences
- `<bos>` (1) - Beginning of sequence
- `<eos>` (2) - End of sequence
- `<unk>` (3) - Unknown/out-of-vocabulary tokens

---

## 🎓 Training

### Training Configuration

```python
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 20
OPTIMIZER = Adam (betas=(0.9, 0.98), eps=1e-9)
LOSS_FUNCTION = CrossEntropyLoss (ignore padding)
GRADIENT_CLIPPING = 1.0
```

### Training Process

1. **Teacher Forcing** - Use ground truth tokens as decoder input
2. **Gradient Clipping** - Prevent exploding gradients (max norm = 1.0)
3. **Validation** - Evaluate after each epoch
4. **Best Model Selection** - Save checkpoint with highest validation BLEU
5. **Early Stopping** - Optional based on validation metrics

### Training Time

- **Kaggle GPU (P100)**: ~2-3 hours for 20 epochs
- **Local GPU (RTX 3080)**: ~1-2 hours for 20 epochs
- **CPU**: ~8-12 hours for 20 epochs (not recommended)

### Training Progress

The model showed steady improvement during training:
- **Epoch 1**: Val BLEU = 0.0149, Val Perplexity = 53.06
- **Epoch 7** (Best): Val BLEU = 0.0357, Val Perplexity = 36.30
- **Epoch 20**: Val BLEU = 0.0289, Val Perplexity = 43.27

Training loss decreased from 4.45 to 2.83, showing good learning progression. The model converged around epoch 7, with later epochs showing signs of overfitting.

---

## 📈 Evaluation Metrics

### Automatic Metrics

1. **BLEU Score** - Measures n-gram overlap with reference
   - Industry standard for machine translation
   - Range: 0.0 to 1.0 (higher is better)

2. **ROUGE-L** - Longest common subsequence F-score
   - Captures sentence-level similarity
   - Better for dialogue than strict BLEU

3. **chrF** - Character n-gram F-score
   - More robust to morphological variations
   - Good for handling typos and variations

4. **Perplexity** - Measures model confidence
   - Lower is better
   - Calculated as: `exp(loss)`

### Manual Evaluation

- **Emotional Appropriateness** - Does the response match the emotion?
- **Contextual Relevance** - Does it relate to the situation?
- **Coherence** - Is the response grammatically correct?
- **Empathy Level** - Does it show understanding and care?

---

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click
4. Share public URL

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Google Drive Model Hosting

The app uses `gdown` to download models from Google Drive:

```python
MODEL_GDRIVE_ID = "1u2eNqSznphIaYuzP91uCtN5Wq-61w0X9"
VOCAB_GDRIVE_ID = "1YaBh4ceJ-XbxEdxES5LsodxVrihZ-c0a"
```

**Important**: Set Google Drive permissions to "Anyone with the link"

---

## 📁 Project Structure

```
empathetic-chatbot/
│
├── app.py                          # Main Streamlit application
├── empathetic_chatbot.ipynb        # Training notebook (Jupyter/Kaggle)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── best_transformer_model.pt       # Best model checkpoint (generated)
├── vocabulary.pkl                  # Vocabulary file (generated)
│
└── demo.csv                        # Sample dataset (optional for local testing)
```

### File Descriptions

- **app.py**: Complete Streamlit web application with:
  - Model architecture definitions
  - Inference functions (greedy & beam search)
  - Google Drive download functionality
  - Interactive UI with conversation history

- **empathetic_chatbot.ipynb**: Full training pipeline including:
  - Data preprocessing
  - Vocabulary building
  - Transformer implementation
  - Training loop with teacher forcing
  - Evaluation on test set
  - Model export

- **requirements.txt**: All Python package dependencies

- **best_transformer_model.pt**: PyTorch checkpoint containing:
  - Model state dictionary
  - Optimizer state
  - Configuration
  - Training metrics

- **vocabulary.pkl**: Pickled vocabulary containing:
  - token2idx mapping
  - idx2token mapping
  - Special token definitions

---

## 📊 Results

### Test Set Performance

| Metric | Score |
|--------|-------|
| **Test Loss** | 3.5711 |
| **Test Perplexity** | 35.56 |
| **Test BLEU** | 0.0261 |
| **Test ROUGE-L** | 0.1507 |
| **Test chrF** | 0.0677 |

### Training Summary

| Metric | Value |
|--------|-------|
| **Total Epochs** | 20 |
| **Best Epoch** | 7 |
| **Best Val BLEU** | 0.0357 |
| **Final Train Loss** | 2.8326 |
| **Final Val Loss** | 3.7674 |

### Example Outputs

#### Example 1: Sentimental

```
Emotion: sentimental
Situation: I remember going to the fireworks with my best friend
Customer: This was a best friend. I miss her.
Agent (Model): i'm so sorry to hear that. what happened?
```

#### Example 2: Afraid

```
Emotion: afraid
Situation: I used to scare for darkness
Customer: it feels like hitting to blank wall when I see the darkness
Agent (Model): that's too bad
```

#### Example 3: Joyful

```
Emotion: joyful
Situation: I got promoted at work today
Customer: I am so happy about this news!
Agent (Model): that's awesome! what are you going to do?
```

---

## 🔮 Future Improvements

### Model Enhancements

- [ ] Increase model size (more layers, larger embeddings)
- [ ] Implement learning rate scheduling
- [ ] Add attention visualization
- [ ] Experiment with different positional encodings
- [ ] Try nucleus sampling (top-p) decoding

### Training Improvements

- [ ] Implement early stopping
- [ ] Add data augmentation
- [ ] Use mixed precision training (FP16)
- [ ] Implement warmup learning rate schedule
- [ ] Add regularization techniques

### Feature Additions

- [ ] Multi-turn conversation support
- [ ] Personality customization
- [ ] Voice input/output
- [ ] Export conversations as PDF
- [ ] User feedback collection
- [ ] Fine-tuning on custom datasets

### Deployment

- [ ] Docker containerization
- [ ] REST API with FastAPI
- [ ] Model quantization for faster inference
- [ ] Caching frequently generated responses
- [ ] Load balancing for multiple users

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Google Drive Download Fails

**Error**: `Failed to retrieve file url`

**Solution**:
- Ensure Google Drive files are set to "Anyone with the link"
- Verify file IDs are correct
- Check internet connection

#### 2. Out of Memory Error

**Error**: `CUDA out of memory`

**Solutions**:
- Reduce `BATCH_SIZE` in config
- Use gradient accumulation
- Train on CPU (slower)
- Use a machine with more GPU memory

#### 3. Model Not Loading

**Error**: `Error loading model`

**Solutions**:
- Check file paths are correct
- Ensure `vocabulary.pkl` and `best_transformer_model.pt` are present
- Verify PyTorch version compatibility
- Try re-downloading model files

#### 4. Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

---

## 📚 References

### Papers

1. **Attention Is All You Need** (Vaswani et al., 2017)
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
   - Original Transformer architecture

2. **Towards Empathetic Open-domain Conversation Models** (Rashkin et al., 2019)
   - [arXiv:1811.00207](https://arxiv.org/abs/1811.00207)
   - Empathetic Dialogues dataset paper

3. **BLEU: a Method for Automatic Evaluation** (Papineni et al., 2002)
   - Standard MT evaluation metric

### Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention Mechanism Explained](https://lilianweng.github.io/posts/2018-06-24-attention/)

---

## 👨‍💻 Author

**Your Name**  
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Facebook AI Research** - For the Empathetic Dialogues dataset
- **Vaswani et al.** - For the Transformer architecture
- **PyTorch Team** - For the excellent deep learning framework
- **Streamlit** - For the easy-to-use web app framework

---

## 📞 Contact & Support

If you have questions or need help:

1. **Open an Issue** - Use GitHub Issues for bug reports
2. **Discussions** - Use GitHub Discussions for questions
3. **Email** - Contact the author directly

---

## ⭐ Star This Repository

If you found this project helpful, please give it a star! ⭐

---

**Last Updated**: October 2025  
**Version**: 1.0.0
