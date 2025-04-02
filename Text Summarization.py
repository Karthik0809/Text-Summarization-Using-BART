# **Install Required Libraries**
"""

pip install transformers datasets torch rouge-score nltk

"""

# **Load Dataset and Tokenizer**"""

from datasets import load_dataset
from transformers import AutoTokenizer

# Load CNN/DailyMail dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def preprocess_function(examples):
    inputs = [doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

from datasets import DatasetDict

dataset = DatasetDict({
    "train": tokenized_datasets["train"],
    "test": tokenized_datasets["test"],
    "validation": tokenized_datasets["validation"] if "validation" in tokenized_datasets else None
})

split_dataset = dataset["train"].train_test_split(test_size=0.1)

tokenized_datasets = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"],
    "test": dataset["test"]
})

"""# **Load Pretrained Model (BART-Large-CNN)**"""

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

from transformers import DataCollatorForSeq2Seq

# Define Data Collator for Sequence-to-Sequence Tasks
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

from transformers import TrainingArguments

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=500,
    report_to="tensorboard"
)

from transformers import Trainer

train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Training Model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

"""# **Train the Model**"""

trainer.train()

import torch

def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

sample_text = dataset["test"][0]["article"]
print("Generated Summary:", generate_summary(sample_text))

"""# **ROUGE Metric for Evaluation**"""

from datasets import load_metric

rouge = load_metric("rouge")

def compute_rouge(predictions, references):
    scores = rouge.compute(predictions=predictions, references=references)
    return scores

# Example Evaluation
pred_summaries = [generate_summary(article) for article in dataset["test"]["article"][:10]]
ref_summaries = dataset["test"]["highlights"][:10]

rouge_scores = compute_rouge(pred_summaries, ref_summaries)
print(rouge_scores)

# Fine-tune Learning Rate and Retrain the Model
training_args.learning_rate = 1e-5
trainer.train()

"""# **Save Fine-Tuned Model and Tokenizer**"""

model.save_pretrained("./fine_tuned_bart")
tokenizer.save_pretrained("./fine_tuned_bart")

# Load the fine-tuned model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("./fine_tuned_bart")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_bart")

