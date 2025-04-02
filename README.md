# Text Summarization using BART

This project fine-tunes the BART (Bidirectional and Auto-Regressive Transformers) model for text summarization using the `facebook/bart-large-cnn` architecture. The model is trained on a dataset of text documents to generate concise and meaningful summaries. BART is a transformer-based sequence-to-sequence model that is particularly effective for summarization tasks, as it combines the strengths of both bidirectional (like BERT) and autoregressive (like GPT) architectures.

---

## Installation

To set up the environment for training and evaluating the BART model, install the required dependencies:

### Clone the repository:
```bash
git clone https://github.com/your-username/abstractive-summarization.git
cd abstractive-summarization
```

### Install dependencies:
```bash
pip install transformers datasets torch rouge-score nltk
```

### Verify installation:
```bash
python -c "import transformers; print(transformers.__version__)"
```

---

## Dataset

The CNN/DailyMail dataset is used for training the model. This dataset consists of news articles along with their corresponding human-written summaries. The dataset is widely used for evaluating summarization models due to its size and quality. It includes:

- **Training Set**: Approximately 287,000 news articles with summaries.
- **Validation Set**: Around 13,000 articles for model tuning.
- **Test Set**: Around 11,000 articles for final evaluation.

The dataset is preprocessed by tokenizing the text and truncating it to a suitable length for BART. It is formatted as `DatasetDict`, ensuring easy handling during training and evaluation.

---

## Model Training

The `facebook/bart-large-cnn` model is fine-tuned using the CNN/DailyMail dataset. The training process involves:

### Data Preprocessing:
- Tokenization using a pretrained BART tokenizer.
- Padding and truncation to fit model input size.
- Data augmentation for better generalization.

### Training Configuration:
- **Batch Size**: 8
- **Epochs**: 3
- **Learning Rate**: 1e-5
- **Optimizer**: AdamW
- **Loss Function**: Cross-entropy

### Training Process:
- The dataset is fed into the BART model using Hugging Face's `Trainer` API.
- The model learns to generate summaries based on input articles.
- Checkpoints are saved after each epoch.
- The model is trained for several epochs to optimize its summarization quality while preventing overfitting.

---

## Evaluation

The fine-tuned model is evaluated using the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric, which compares the generated summaries with the reference summaries. The key ROUGE scores include:

- **ROUGE-1**: Measures the overlap of individual words between generated and reference summaries.
- **ROUGE-2**: Evaluates the overlap of bigrams (two-word sequences), providing insight into fluency and coherence.
- **ROUGE-L**: Considers the longest common subsequence (LCS) to measure structural similarity.

These metrics provide a robust assessment of summarization quality. Higher scores indicate better summarization performance.

---

## Usage

The trained BART model can generate concise and meaningful summaries from long-form text. Given an input article, the model produces a summary that captures the key points while maintaining fluency. The summarization process involves:

1. **Preprocessing the Input**: Tokenizing and truncating the input text.
2. **Generating the Summary**: Using beam search decoding to ensure coherence.
3. **Postprocessing**: Removing special tokens and formatting the output for readability.

### Applications:
- **News summarization tools.**
- **Content summarization for research papers.**
- **Automated report generation for businesses.**

---

## Example Input and Output:

### Input Article:
```plaintext
A team of scientists discovered a new exoplanet in the habitable zone of a distant star. The planet, named Kepler-442b, has conditions that may support life. Researchers believe it has a stable atmosphere and liquid water on its surface. Further studies will be conducted to analyze its composition.
```

### Generated Summary:
```plaintext
Scientists have found an exoplanet, Kepler-442b, in a distant star's habitable zone. It may support life, with a stable atmosphere and liquid water.
```

---

## Results

After fine-tuning, the model achieves the following ROUGE scores on the validation set:

| Metric  | Score  |
|---------|--------|
| ROUGE-1 | 44.8   |
| ROUGE-2 | 21.3   |
| ROUGE-L | 41.6   |

These results indicate that the model performs well in summarizing text while preserving key information.

---

## Future Improvements

To enhance the performance further, potential improvements include:

- **Using a larger dataset** for better generalization.
- **Experimenting with reinforcement learning** to fine-tune summary coherence.
- **Incorporating knowledge distillation** to optimize model efficiency.

---

## References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [BART Model Paper](https://arxiv.org/abs/1910.13461)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail)
