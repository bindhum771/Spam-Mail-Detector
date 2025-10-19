Spam-Mail-Detector
🚀 Next-Level AI Spam Filter (DistilBERT + XAI)

This project implements a state-of-the-art spam and phishing detector using **DistilBERT**, a compact version of the BERT transformer model. It is deployed as an interactive web application using **Streamlit**, which also includes an **Explainable AI (XAI)** feature to highlight the words the model pays the most attention to when making a decision.

The core functionality involves fine-tuning DistilBERT for a sequence classification task (Spam vs. Ham). If a full `spam.csv` dataset is not found, the application uses a small dummy dataset to ensure the app runs immediately.

***

✨ Features

* **Transformer-Based Classification:** Uses the powerful pre-trained **DistilBERT** model for high-accuracy text classification.
* **Streamlit Web App:** Provides an interactive and easy-to-use interface for real-time message classification.
* **Explainable AI (XAI):** Visualizes the model's decision-making process by highlighting the most influential words using **attention scores** (Red for Spam, Blue for Ham).
* **On-the-Fly Training:** Automatically trains the model on the first run using the provided dataset.
* **Performance Metrics:** Displays a **Confusion Matrix** to show the model's performance on a held-out test set.
* **Persistence:** Uses Streamlit's resource caching and file saving to store the trained model, tokenizer, and metrics (`./bert_model_output`), significantly speeding up subsequent app loads.

 ⚙️ Key Technologies

* **Python:** Programming Language
* **Streamlit:** Application Framework
* **PyTorch:** Deep Learning Library
* **Hugging Face Transformers:** Used for DistilBERT implementation
* **DistilBERT:** Transformer Model for Sequence Classification
* **XAI (Attention Weights):** Used for highlighting important words

***

💡 How the XAI Works

The **Explainable AI (XAI)** feature utilizes the **attention weights** from the last layer of the DistilBERT model to justify its prediction.

1.  **Attention Scores:** When a message is classified, the model generates attention scores for every token (word or sub-word). We focus on the scores related to the special `[CLS]` (Classification) token, as it aggregates the information for the final decision.
2.  **Visualization:** The words are visually rendered in the app, with the **intensity of the color** (Red for Spam/Phishing, Blue for Ham) corresponding to how much attention the model paid to that word.
3.  **Explanation:** This provides an immediate, intuitive explanation for the AI's prediction, showing **which words** made the model think the message was spam (e.g., 'URGENT', 'CLAIM', 'FREE', 'link').

***Output Screenshots
<img width="1366" height="768" alt="Screenshot (240)" src="https://github.com/user-attachments/assets/05465f14-8ced-48ab-8781-8d8d68933e0b" />
<img width="1366" height="768" alt="Screenshot (241)" src="https://github.com/user-attachments/assets/5ec58326-d247-466a-95df-458c61866c87" />
<img width="1366" height="768" alt="Screenshot (242)" src="https://github.com/user-attachments/assets/d720ec02-bc80-415c-8943-c5ebbe62cef8" />

🛠️ Installation and Setup

 1. Prerequisites

You need **Python 3.8+** installed.

2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
Activate the environment
On Windows:
\venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate



