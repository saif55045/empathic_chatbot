"""
Empathetic Chatbot - Streamlit App (with model download)
A Transformer-based chatbot that generates empathetic responses

This version downloads the model from Google Drive if not present locally
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import math
import re
import os
import gdown

# ============================================
# Config Class (needed for checkpoint loading)
# ============================================

class Config:
    """Configuration class for model hyperparameters"""
    DATA_PATH = '/kaggle/input/empathetic-dialogues-facebook-ai/demo.csv'
    EMBEDDING_DIM = 256
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    FFN_DIM = 512
    DROPOUT = 0.1
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 20
    GRAD_CLIP = 1.0
    PAD_TOKEN = '<pad>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'
    UNK_TOKEN = '<unk>'
    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 10000
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

# Set page config
st.set_page_config(
    page_title="Empathetic Chatbot",
    page_icon="🤗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# File Download Configuration
# ============================================

# IMPORTANT: Replace these with your own Google Drive file IDs
# After uploading to Google Drive, get shareable link and extract the FILE_ID
# Link format: https://drive.google.com/file/d/FILE_ID/view?usp=sharing

MODEL_GDRIVE_ID = "1u2eNqSznphIaYuzP91uCtN5Wq-61w0X9"  # Just the file ID
VOCAB_GDRIVE_ID = "1YaBh4ceJ-XbxEdxES5LsodxVrihZ-c0a"  # Just the file ID

MODEL_FILE = "best_transformer_model.pt"
VOCAB_FILE = "vocabulary.pkl"

def download_from_gdrive(file_id, output_path):
    """Download file from Google Drive"""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)

def ensure_files_exist():
    """Download model files if they don't exist"""
    files_to_download = []
    
    if not os.path.exists(MODEL_FILE):
        if MODEL_GDRIVE_ID != "YOUR_MODEL_FILE_ID_HERE":
            files_to_download.append((MODEL_GDRIVE_ID, MODEL_FILE, "Model"))
        else:
            st.error(f"❌ {MODEL_FILE} not found. Please configure Google Drive file IDs or place the file in the app directory.")
            return False
    
    if not os.path.exists(VOCAB_FILE):
        if VOCAB_GDRIVE_ID != "YOUR_VOCAB_FILE_ID_HERE":
            files_to_download.append((VOCAB_GDRIVE_ID, VOCAB_FILE, "Vocabulary"))
        else:
            st.error(f"❌ {VOCAB_FILE} not found. Please configure Google Drive file IDs or place the file in the app directory.")
            return False
    
    if files_to_download:
        st.info("📥 Downloading model files from Google Drive...")
        progress_bar = st.progress(0)
        
        for idx, (file_id, output_path, name) in enumerate(files_to_download):
            st.write(f"Downloading {name}...")
            try:
                download_from_gdrive(file_id, output_path)
                progress_bar.progress((idx + 1) / len(files_to_download))
            except Exception as e:
                st.error(f"Failed to download {name}: {e}")
                return False
        
        st.success("✅ Files downloaded successfully!")
    
    return True

# ============================================
# Model Architecture (Same as training)
# ============================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(x)
        
        return output, attention_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Transformer Decoder Layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer Encoder-Decoder Model"""
    
    def __init__(self, vocab_size, d_model=256, num_heads=4, 
                 num_encoder_layers=2, num_decoder_layers=2, 
                 d_ff=512, dropout=0.1, max_seq_len=128, pad_idx=0):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.encoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        batch_size, tgt_len = tgt.size()
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def encode(self, src, src_mask):
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        output = self.fc_out(decoder_output)
        
        return output


# ============================================
# Vocabulary Class
# ============================================

class Vocabulary:
    """Vocabulary class for token to index mapping"""
    
    def __init__(self, token2idx, idx2token, pad_token='<pad>', 
                 bos_token='<bos>', eos_token='<eos>', unk_token='<unk>'):
        self.token2idx = token2idx
        self.idx2token = idx2token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
    def encode(self, text, add_special_tokens=True):
        """Convert text to token indices"""
        tokens = self.simple_tokenize(self.normalize_text(text))
        indices = []
        
        if add_special_tokens:
            indices.append(self.token2idx[self.bos_token])
        
        for token in tokens:
            indices.append(self.token2idx.get(token, self.token2idx[self.unk_token]))
        
        if add_special_tokens:
            indices.append(self.token2idx[self.eos_token])
        
        return indices
    
    def decode(self, indices, skip_special_tokens=True):
        """Convert token indices back to text"""
        tokens = []
        special_tokens = {self.token2idx[self.pad_token], 
                         self.token2idx[self.bos_token], 
                         self.token2idx[self.eos_token]}
        
        for idx in indices:
            if skip_special_tokens and idx in special_tokens:
                continue
            tokens.append(self.idx2token.get(idx, self.unk_token))
        
        return ' '.join(tokens)
    
    @staticmethod
    def normalize_text(text):
        """Normalize text"""
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    @staticmethod
    def simple_tokenize(text):
        """Simple word tokenization"""
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def __len__(self):
        return len(self.token2idx)


# ============================================
# Inference Functions
# ============================================

def greedy_decode(model, src, vocab, max_len=50, device='cpu'):
    """Greedy decoding for inference"""
    model.eval()
    
    with torch.no_grad():
        src = src.to(device)
        src_mask = model.make_src_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        tgt_indices = [vocab.token2idx[vocab.bos_token]]
        
        for _ in range(max_len):
            tgt = torch.LongTensor([tgt_indices]).to(device)
            tgt_mask = model.make_tgt_mask(tgt)
            
            decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
            output = model.fc_out(decoder_output)
            
            next_token = output[0, -1, :].argmax().item()
            tgt_indices.append(next_token)
            
            if next_token == vocab.token2idx[vocab.eos_token]:
                break
        
        return tgt_indices


def beam_search_decode(model, src, vocab, beam_width=3, max_len=50, device='cpu'):
    """Beam search decoding for inference"""
    model.eval()
    
    with torch.no_grad():
        src = src.to(device)
        src_mask = model.make_src_mask(src)
        encoder_output = model.encode(src, src_mask)
        
        beams = [([vocab.token2idx[vocab.bos_token]], 0.0)]
        
        for _ in range(max_len):
            candidates = []
            
            for seq, score in beams:
                if seq[-1] == vocab.token2idx[vocab.eos_token]:
                    candidates.append((seq, score))
                    continue
                
                tgt = torch.LongTensor([seq]).to(device)
                tgt_mask = model.make_tgt_mask(tgt)
                
                decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
                output = model.fc_out(decoder_output)
                
                log_probs = F.log_softmax(output[0, -1, :], dim=0)
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for prob, idx in zip(top_k_probs, top_k_indices):
                    new_seq = seq + [idx.item()]
                    new_score = score + prob.item()
                    candidates.append((new_seq, new_score))
            
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            if all(seq[-1] == vocab.token2idx[vocab.eos_token] for seq, _ in beams):
                break
        
        best_seq, best_score = beams[0]
        return best_seq


def generate_response(model, input_text, vocab, device='cpu', method='greedy', beam_width=3):
    """Generate response for given input text"""
    model.eval()
    
    src_indices = vocab.encode(input_text, add_special_tokens=True)
    src = torch.LongTensor([src_indices])
    
    if method == 'greedy':
        output_indices = greedy_decode(model, src, vocab, device=device)
    elif method == 'beam':
        output_indices = beam_search_decode(model, src, vocab, beam_width=beam_width, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    output_text = vocab.decode(output_indices, skip_special_tokens=True)
    
    return output_text


# ============================================
# Load Model and Vocabulary
# ============================================

@st.cache_resource
def load_model_and_vocab():
    """Load the trained model and vocabulary"""
    # Ensure files exist (download if needed)
    if not ensure_files_exist():
        st.stop()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    with open(VOCAB_FILE, 'rb') as f:
        vocab_data = pickle.load(f)
    
    vocab = Vocabulary(
        token2idx=vocab_data['token2idx'],
        idx2token=vocab_data['idx2token'],
        pad_token=vocab_data['pad_token'],
        bos_token=vocab_data['bos_token'],
        eos_token=vocab_data['eos_token'],
        unk_token=vocab_data['unk_token']
    )
    
    # Load model - use weights_only=False for compatibility with PyTorch 2.6+
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('config', None)
    
    # Extract configuration - handle both dict and object formats
    config_values = {}
    if model_config:
        if hasattr(model_config, '__dict__'):
            config_dict = model_config.__dict__
        elif isinstance(model_config, dict):
            config_dict = model_config
        else:
            config_dict = {}
        
        config_values = {
            'd_model': config_dict.get('EMBEDDING_DIM', config_dict.get('d_model', 256)),
            'num_heads': config_dict.get('NUM_HEADS', config_dict.get('num_heads', 4)),
            'num_encoder_layers': config_dict.get('NUM_ENCODER_LAYERS', config_dict.get('num_encoder_layers', 2)),
            'num_decoder_layers': config_dict.get('NUM_DECODER_LAYERS', config_dict.get('num_decoder_layers', 2)),
            'd_ff': config_dict.get('FFN_DIM', config_dict.get('d_ff', 512)),
            'dropout': config_dict.get('DROPOUT', config_dict.get('dropout', 0.1)),
            'max_seq_len': config_dict.get('MAX_SEQ_LEN', config_dict.get('max_seq_len', 128))
        }
    else:
        config_values = {
            'd_model': 256,
            'num_heads': 4,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'd_ff': 512,
            'dropout': 0.1,
            'max_seq_len': 128
        }
    
    # Create model
    model = Transformer(
        vocab_size=len(vocab),
        d_model=config_values['d_model'],
        num_heads=config_values['num_heads'],
        num_encoder_layers=config_values['num_encoder_layers'],
        num_decoder_layers=config_values['num_decoder_layers'],
        d_ff=config_values['d_ff'],
        dropout=config_values['dropout'],
        max_seq_len=config_values['max_seq_len'],
        pad_idx=vocab.token2idx[vocab.pad_token]
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, device


# ============================================
# Streamlit App (Rest of the UI code - same as before)
# ============================================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .user-message {
            background-color: #e3f2fd;
            border-left: 5px solid #1f77b4;
        }
        .bot-message {
            background-color: #f1f8e9;
            border-left: 5px solid #4caf50;
        }
        .emotion-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            font-size: 0.9rem;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🤗 Empathetic Chatbot</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #555;">A Transformer-based chatbot trained to generate empathetic responses</p>',
        unsafe_allow_html=True
    )
    
    # Load model
    try:
        model, vocab, device = load_model_and_vocab()
        st.sidebar.success(f"✅ Model loaded on {device}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    decoding_method = st.sidebar.radio(
        "Decoding Strategy",
        ["Greedy", "Beam Search"],
        help="Choose the decoding strategy for response generation"
    )
    
    beam_width = 3
    if decoding_method == "Beam Search":
        beam_width = st.sidebar.slider(
            "Beam Width",
            min_value=2,
            max_value=5,
            value=3,
            help="Number of beams for beam search"
        )
    
    # Emotion selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎭 Select Emotion")
    emotions = [
        "sentimental", "afraid", "proud", "faithful", "terrified",
        "joyful", "angry", "sad", "jealous", "grateful",
        "prepared", "embarrassed", "excited", "annoyed", "lonely",
        "ashamed", "guilty", "surprised", "nostalgic", "confident",
        "furious", "disappointed", "caring", "trusting", "disgusted",
        "anticipating", "anxious", "hopeful", "content", "impressed",
        "apprehensive", "devastated"
    ]
    
    selected_emotion = st.sidebar.selectbox(
        "Emotion",
        emotions,
        index=0
    )
    
    # Main area
    st.markdown("---")
    
    # Initialize session state for conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Input form
    with st.form(key='input_form', clear_on_submit=True):
        col1, col2 = st.columns([1, 1])
        
        with col1:
            situation = st.text_area(
                "Situation/Context:",
                placeholder="Describe the situation or context...",
                height=100,
                help="Provide context about the situation"
            )
        
        with col2:
            customer_message = st.text_area(
                "Your Message:",
                placeholder="What would you like to say?",
                height=100,
                help="Your message or question"
            )
        
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_b:
            submit_button = st.form_submit_button("💬 Generate Response", use_container_width=True)
    
    # Generate response
    if submit_button:
        if not situation.strip() or not customer_message.strip():
            st.warning("⚠️ Please fill in both situation and message fields.")
        else:
            with st.spinner("🤔 Generating empathetic response..."):
                input_text = f"Emotion: {selected_emotion} | Situation: {situation} | Customer: {customer_message} Agent:"
                
                method = 'greedy' if decoding_method == "Greedy" else 'beam'
                response = generate_response(
                    model, input_text, vocab, 
                    device=device, method=method, beam_width=beam_width
                )
                
                st.session_state.conversation_history.append({
                    'emotion': selected_emotion,
                    'situation': situation,
                    'user_message': customer_message,
                    'bot_response': response
                })
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")
        
        col1, col2, col3 = st.columns([4, 1, 1])
        with col3:
            if st.button("🗑️ Clear History"):
                st.session_state.conversation_history = []
                st.rerun()
        
        for idx, conv in enumerate(reversed(st.session_state.conversation_history)):
            with st.container():
                emotion_color = {
                    'joyful': '#4caf50', 'happy': '#4caf50', 'excited': '#4caf50',
                    'sad': '#2196f3', 'lonely': '#2196f3', 'disappointed': '#2196f3',
                    'angry': '#f44336', 'furious': '#f44336', 'annoyed': '#f44336',
                    'afraid': '#ff9800', 'terrified': '#ff9800', 'anxious': '#ff9800',
                    'surprised': '#9c27b0', 'impressed': '#9c27b0'
                }.get(conv['emotion'], '#607d8b')
                
                st.markdown(
                    f'<span class="emotion-badge" style="background-color: {emotion_color}; color: white;">'
                    f'😊 {conv["emotion"].capitalize()}</span>',
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**📍 Situation:** {conv['situation']}")
                
                st.markdown(
                    f'<div class="chat-message user-message">'
                    f'<strong>👤 You:</strong><br>{conv["user_message"]}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown(
                    f'<div class="chat-message bot-message">'
                    f'<strong>🤖 Agent:</strong><br>{conv["bot_response"]}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
        ### 📊 About This Model
        
        - **Architecture**: Transformer Encoder-Decoder
        - **Training**: Empathetic Dialogues Dataset
        - **Features**:
          - Multi-head attention
          - Positional encoding
          - Teacher forcing
          - Greedy & Beam search decoding
        
        ---
        Built with ❤️ using PyTorch and Streamlit
    """)
    
    # Example inputs
    with st.expander("💡 Example Inputs"):
        st.markdown("""
        **Example 1:**
        - **Emotion**: sentimental
        - **Situation**: I remember going to the fireworks with my best friend
        - **Message**: This was a best friend. I miss her.
        
        **Example 2:**
        - **Emotion**: afraid
        - **Situation**: I used to scare for darkness
        - **Message**: it feels like hitting to blank wall when I see the darkness
        
        **Example 3:**
        - **Emotion**: joyful
        - **Situation**: I got promoted at work today
        - **Message**: I am so happy about this news!
        """)


if __name__ == "__main__":
    main()
