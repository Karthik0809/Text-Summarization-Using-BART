# 📝 Text Summarization Using BART

This project implements an abstractive text summarization system using a fine-tuned [BART](https://arxiv.org/abs/1910.13461) model. The app uses **Streamlit** as the front-end interface and provides a simple UI to summarize custom input text using a pre-trained or fine-tuned BART model.

![image](https://github.com/user-attachments/assets/5e325303-e18b-4602-920f-3bc819eb1044)

---

## 📁 Project Structure

```
summarizer_project/
├── .gitignore
├── fine_tuned_bart/             # Contains tokenizer config files (exclude large model weights)
├── evaluate_rouge.py            # Evaluation script using ROUGE metrics
├── load_model.py                # Loads the summarization model
├── requirements.txt             # List of dependencies
├── setup.bat                    # Optional batch file to setup the environment
├── streamlit_app.py             # Main Streamlit UI app
├── summarize_input.py           # CLI script to summarize a given input text
├── summarizer.py                # Core summarization logic using BART
├── utils.py                     # Helper functions (e.g., text preprocessing)
```

> ⚠️ Note: Model weights (`model.safetensors`) are too large to upload to GitHub. Use HuggingFace Hub or download them manually for local use.

---

## 🛠 Tech Stack

- Python 3.11+
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [Streamlit](https://streamlit.io/)
- ROUGE score evaluation via `evaluate` or `rouge_score`

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Karthik0809/Text-Summarization-Using-BART.git
cd Text-Summarization-Using-BART
```

### 2. Create Virtual Environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate   # macOS/Linux
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

This will launch a browser window where you can input long-form text and receive its summarized version using BART.

---

## 🧪 Evaluating the Model

Use the ROUGE evaluation script:
```bash
python evaluate_rouge.py
```

Modify the script to compare your model output with ground truth summaries as needed.

---

## 📦 Folder Notes

- `fine_tuned_bart/`: Include tokenizer files only (e.g., `merges.txt`, `vocab.json`). Avoid uploading `model.safetensors` or any >100MB files.
- `.venv/` and `venv/`: Not tracked by Git due to `.gitignore`.
- `requirements.txt`: Auto-generated or manually listed Python packages.

---

## 🧠 Model

The model is based on:
- **facebook/bart-large-cnn**: Pre-trained BART model.
- You may fine-tune it further using your custom dataset for domain-specific summarization.

---

## ✍️ Author

**Karthik Mulugu**  
[GitHub](https://github.com/Karthik0809) | [LinkedIn](https://www.linkedin.com/in/karthikmulugu/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
