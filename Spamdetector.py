import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import shutil 

# --- Deep Learning/Transfer Learning Libraries ---
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# --- Configuration ---
MODEL_NAME = 'distilbert-base-uncased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3 

# Persistence Paths
MODEL_DIR = './bert_model_output'
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.pkl')

# Set device for PyTorch (use GPU if available)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --- 1. Custom PyTorch Dataset Class ---
class SpamDataset(Dataset):
    """A standard PyTorch Dataset for handling encoded text data."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# --- 2. Data Loading and BERT Encoding ---
@st.cache_data
def load_data():
    """Loads dataset and prepares labels."""
    try:
        # Tries to load the full dataset if available
        data = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], names=['label', 'message'], header=0)
        data = data.iloc[1:]
    except Exception:
        # Fallback to a small dummy dataset
        data = pd.DataFrame({
            'label': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
            'message': [
                'Hi, are you free for lunch tomorrow?',
                'URGENT! You won $1000 cash prize! Claim NOW at http://fake.com',
                'Thanks for the update on the project.',
                'SECURITY ALERT: Click here immediately to verify your account: www.phish.net',
                'Please review the attached file.',
                'Your FREE entry to the $1M draw has been confirmed. Click NOW!',
                'Got the details, thanks.',
                'Congratulations! You have been selected. Call 1-800-SPAM now.'
            ]
        })

    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    messages = data['message']
    labels = data['label'].values
    return messages, labels


def prepare_bert_data(messages, labels):
    """Encodes text using the BERT tokenizer, splits data using indices, and creates DataLoaders."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Encode all messages for BERT
    encodings = tokenizer(
        messages.tolist(), 
        truncation=True, 
        padding=True, 
        max_length=MAX_LEN
    )
    
    # --- Data Splitting Fix (using indices) ---
    indices = np.arange(len(labels))
    
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices, labels, test_size=0.2, random_state=42, stratify=labels
    )

    X_train_enc = {}
    X_test_enc = {}
    
    for key in encodings.keys():
        X_train_enc[key] = np.array(encodings[key])[train_idx]
        X_test_enc[key] = np.array(encodings[key])[test_idx]

    train_dataset = SpamDataset(X_train_enc, y_train.tolist())
    test_dataset = SpamDataset(X_test_enc, y_test.tolist())
    
    return train_dataset, test_dataset, tokenizer, y_test


# --- 3. Model Fine-Tuning (Training) ---
@st.cache_resource
def train_bert_model(_train_dataset, _test_dataset, y_test, _tokenizer): 
    """Loads, fine-tunes DistilBERT, and saves the model and tokenizer."""
    
    st.info("🧠 Fine-Tuning DistilBERT (AI Model Training)... This will take a few minutes.", icon="🧠")
    
    # Load pre-trained model for sequence classification
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    
    train_loader = DataLoader(_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(_test_dataset, batch_size=BATCH_SIZE)
    
    optimizer = AdamW(model.parameters(), lr=5e-5) 

    # --- Training Loop ---
    model.train()
    for epoch in range(EPOCHS):
        st.caption(f"Epoch {epoch + 1}/{EPOCHS}...")
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()

    # --- Evaluation and Saving ---
    model.eval()
    all_preds = []
    
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(predictions)

    cm = confusion_matrix(y_test, all_preds)

    # Save components
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    model.save_pretrained(MODEL_DIR) 
    _tokenizer.save_pretrained(MODEL_DIR) 
    
    joblib.dump(cm, METRICS_PATH)
    
    st.success("DistilBERT model fine-tuning and saving complete!")
    return model, cm, _tokenizer 


# --- 4. Model Loading and Prediction ---

def load_bert_model():
    """Loads the model, CM, and tokenizer from disk, or initiates training."""
    
    # 1. Try to load saved components (Model, CM, and Tokenizer)
    try:
        if os.path.exists(MODEL_DIR) and os.path.exists(METRICS_PATH):
            st.info("Loading fine-tuned AI model from disk...", icon="💾")
            
            model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
            model.to(DEVICE)
            model.eval() 
            cm = joblib.load(METRICS_PATH)
            tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR) 
            
            return model, cm, tokenizer
        
    except OSError as e:
        # 2. If load fails (e.g., corrupted files), warn user and force training.
        st.error(f"Error loading saved model/tokenizer files from {MODEL_DIR}. Deleting directory and forcing re-training...")
        
        if os.path.exists(MODEL_DIR):
             shutil.rmtree(MODEL_DIR)
        
        pass 
    
    # 3. Initiate Training
    messages, labels = load_data()
    train_dataset, test_dataset, tokenizer, y_test = prepare_bert_data(messages, labels)
    
    return train_bert_model(train_dataset, test_dataset, y_test, tokenizer)


# --- XAI Explanation Functions (NEW) ---

def visualize_attention(tokens, scores, is_spam):
    """Generates colored HTML text based on token attention scores."""
    
    # Normalize scores from 0 to 1 for coloring. Ignore [CLS] and [SEP] tokens.
    # Scores array includes [CLS] and [SEP] tokens at indices 0 and -1 (or last valid index)
    
    # Find the indices corresponding to real tokens
    valid_indices = [i for i, token in enumerate(tokens) if token not in ['[CLS]', '[SEP]', '[PAD]']]
    
    if not valid_indices:
        return "<p>No detailed word analysis available for this short message.</p>"

    valid_scores = scores[valid_indices]
    valid_tokens = [tokens[i] for i in valid_indices]
    
    # Min-max normalization for coloring strength
    min_score = valid_scores.min()
    max_score = valid_scores.max()
    
    # Avoid division by zero if all scores are identical
    if max_score == min_score:
        normalized_scores = np.ones_like(valid_scores) * 0.5
    else:
        normalized_scores = (valid_scores - min_score) / (max_score - min_score)
        
    html_text = []
    
    # Choose color based on prediction
    color_base = "255, 0, 0" if is_spam else "0, 100, 255" # Red for Spam, Blue for Ham
    
    for token, score in zip(valid_tokens, normalized_scores):
        # Decode special tokens like ##word to 'word'
        display_token = token.replace('##', '')
        
        # Scale opacity (alpha) from 0.2 to 1.0
        opacity = 0.2 + 0.8 * score 
        
        # Apply CSS for highlighting
        style = f"background-color: rgba({color_base}, {opacity}); border-radius: 3px; padding: 2px 0;"
        
        html_text.append(f'<span style="{style}">{display_token}</span>')

    return f'<div style="line-height: 2.0; font-size: 1.1em; margin-top: 15px;">{" ".join(html_text)}</div>'

def bert_predict_explain(text, tokenizer, model):
    """
    Performs inference and returns prediction, confidence, and attention scores for XAI.
    """
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_LEN
    )
    
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)
    
    with torch.no_grad():
        # Request attention weights from the model
        outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)
    
    probabilities = torch.softmax(outputs.logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    spam_proba = probabilities[0][1].item()
    
    # --- Attention Extraction ---
    # outputs.attentions is a tuple (num_layers) of tensors 
    # Take the attention from the last layer (most contextualized)
    last_layer_att = outputs.attentions[-1][0] 
    
    # Average attention across all heads (dim=0), looking at the [CLS] token (index 0 in dim=2)
    att_scores = last_layer_att.mean(dim=0)[:, 0].cpu().numpy()
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return predicted_class_id, spam_proba, tokens, att_scores


# --- 5. Visualization (CM) ---
def plot_confusion_matrix(cm):
    """Generates a matplotlib figure for the confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = ['HAM', 'SPAM']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('AI Model Performance (DistilBERT Test Data)')
    plt.close(fig)
    return fig


# --- Main Application Logic ---

# Load or train the components (model, confusion matrix, and tokenizer)
model, confusion_mat, tokenizer = load_bert_model()

# --- 6. Streamlit User Interface ---
st.title("🚀 Next-Level AI Spam Filter (DistilBERT)")
st.markdown("### Powered by Transfer Learning and Explainable AI (XAI)")

# Input Section
st.subheader("Simulated Email Message")
user_input = st.text_area(
    "Enter the Email or SMS message to classify:",
    "URGENT: Your bank account has been locked. Click this link immediately to restore access: https://secure-bank-login.net",
    height=150
)

# Classification Button
if st.button("Classify Message (AI Prediction)", type="primary"):
    if user_input:
        try:
            # CALL THE NEW EXPLAINABLE FUNCTION
            predicted_class, spam_confidence, tokens, att_scores = bert_predict_explain(user_input, tokenizer, model)

            st.markdown("---")
            st.subheader("Classification Status")
            
            is_spam = (predicted_class == 1)

            if is_spam:
                st.error("🚫 CLASSIFIED AS **SPAM/PHISHING**!")
                st.caption(f"The AI model is **{spam_confidence*100:.2f}%** confident this is deceptive mail.")
            else:
                ham_confidence = 1 - spam_confidence
                st.success("✅ CLASSIFIED AS **HAM** (Not Spam).")
                st.caption(f"The AI model is **{ham_confidence*100:.2f}%** confident this message is safe.")
            
            
            # --- XAI EXPLANATION SECTION (NEW) ---
            st.subheader("🤖 AI Decision Explanation (Attention)")
            st.markdown(f"The model paid the most **attention** to the words highlighted in **{ 'red (Spam)' if is_spam else 'blue (Ham)'}** to make its decision.")
            
            # Display the visualization
            explanation_html = visualize_attention(tokens, att_scores, is_spam)
            st.markdown(explanation_html, unsafe_allow_html=True)
            
            
        except Exception as e:
            st.error(f"An error occurred during classification: {e}")
            st.warning("Ensure all libraries are installed and the model has finished initial training.")
    else:
        st.warning("Please enter a message to classify.")

# --- 7. Model Evaluation Graph ---
st.markdown("---")
st.subheader("AI Model Performance Graph (BERT Test Data)")

if confusion_mat is not None:
    st.pyplot(plot_confusion_matrix(confusion_mat))
    
    st.caption("""
    This graph shows the DistilBERT model's performance. The goal of a professional filter is to minimize **False Positives (FP)**—Ham emails incorrectly flagged as Spam.
    """)