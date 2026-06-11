import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ─── 1. DEFINE REPAIR CORPUS ───────────────────────────────────────────────
repair_corpus = pd.DataFrame({
    "doc_id": ["DOC-001", "DOC-002", "DOC-003", "DOC-004", "DOC-005"],
    "component": ["spindle motor", "hydraulic pump", "drive belt", "bearing assembly", "hydraulic pump"],
    "failure_mode": ["overheating", "temperature spike", "mechanical fracture", "vibration", "voltage spike"],
    "manual_text": [
        "To resolve severe overheating in the spindle motor, replace the motor thermal paste, verify air filter clearance, and check the ventilation shaft. Estimated downtime: 2 hours.",
        "For abnormal temperature spikes or thermal issues in the hydraulic pump, flush the coolant system and recalibrate the thermal sensors. Estimated downtime: 3 hours.",
        "A mechanical fracture in the drive belt requires a complete replacement of the Poly-V belt and recalibration of the tensioner pulley. Estimated downtime: 4 hours.",
        "Excessive vibration in the bearing assembly should be treated by applying industrial lubricant. If pitting is visible, replace the bearing kit. Estimated downtime: 1.5 hours.",
        "Voltage spikes in the hydraulic pump trigger the emergency breaker. Reset the breaker and inspect the wiring harness for arc damage. Estimated downtime: 1 hour."
    ],
    "parts_required": ["Thermal Paste, Air Filter", "Coolant Fluid", "Poly-V Belt", "Industrial Lubricant", "Wiring Harness"]
})

# ─── 2. TF-IDF RETRIEVER ──────────────────────────────────────────────────
class TfidfRetriever:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.corpus_matrices = self.tfidf.fit_transform(corpus['manual_text'])
        
    def get_relevant_context(self, operator_log, top_k=2):
        query_vec = self.tfidf.transform([operator_log])
        similarities = cosine_similarity(query_vec, self.corpus_matrices).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        contexts = [self.corpus.iloc[idx]['manual_text'] for idx in top_indices]
        return " ".join(contexts)

# ─── 3. LOAD DATA ──────────────────────────────────────────────────────────
print("Loading train, val, and test splits...")
df_train = pd.read_csv("data/train_logs.csv")
df_val = pd.read_csv("data/val_logs.csv")
df_test = pd.read_csv("data/test_logs.csv")

retriever = TfidfRetriever(repair_corpus)

def enrich_dataset(df):
    enriched_texts = []
    for idx, row in df.iterrows():
        context = retriever.get_relevant_context(row['log_text'], top_k=2)
        enriched = f"Operator Log: {row['log_text']}\nManual Context: {context}"
        enriched_texts.append(enriched)
    df['enriched_text'] = enriched_texts
    return df

print("Enriching splits with TF-IDF manual contexts...")
df_train = enrich_dataset(df_train)
df_val = enrich_dataset(df_val)
df_test = enrich_dataset(df_test)

# ─── 4. TOKENIZE ───────────────────────────────────────────────────────────
print("Tokenizing datasets...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(df_train['enriched_text']), truncation=True, padding='max_length', max_length=256)
val_encodings = tokenizer(list(df_val['enriched_text']), truncation=True, padding='max_length', max_length=256)
test_encodings = tokenizer(list(df_test['enriched_text']), truncation=True, padding='max_length', max_length=256)

class MaintenanceDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Datasets for Component Classifier
train_dataset_comp = MaintenanceDataset(train_encodings, list(df_train['component_id']))
val_dataset_comp = MaintenanceDataset(val_encodings, list(df_val['component_id']))
test_dataset_comp = MaintenanceDataset(test_encodings, list(df_test['component_id']))

# Datasets for Failure Mode Classifier
train_dataset_fail = MaintenanceDataset(train_encodings, list(df_train['failure_mode_id']))
val_dataset_fail = MaintenanceDataset(val_encodings, list(df_val['failure_mode_id']))
test_dataset_fail = MaintenanceDataset(test_encodings, list(df_test['failure_mode_id']))

# ─── 5. TRAINING HELPERS ──────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# output_dir for checkpoints, we can set unique ones for each model to avoid collision
training_args_comp = TrainingArguments(
    output_dir='./models/tier3_comp_checkpoints',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    warmup_steps=50,
    logging_dir='./logs',
    report_to='none'
)

training_args_fail = TrainingArguments(
    output_dir='./models/tier3_fail_checkpoints',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    warmup_steps=50,
    logging_dir='./logs',
    report_to='none'
)

# ─── 6. TRAIN COMPONENT CLASSIFIER ─────────────────────────────────────────
print("\n" + "="*70)
print("Training Model A: Component Classifier...")
print("="*70)
model_component = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=5
)

# Enable MPS acceleration if available on macOS
device = "mps" if torch.backends.mps.is_available() else "cpu"
model_component.to(device)
print(f"Using device: {device}")

trainer_comp = Trainer(
    model=model_component,
    args=training_args_comp,
    train_dataset=train_dataset_comp,
    eval_dataset=val_dataset_comp,
    compute_metrics=compute_metrics
)
trainer_comp.train()

# ─── 7. TRAIN FAILURE MODE CLASSIFIER ──────────────────────────────────────
print("\n" + "="*70)
print("Training Model B: Failure Mode Classifier...")
print("="*70)
model_failure = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=5
)
model_failure.to(device)

trainer_fail = Trainer(
    model=model_failure,
    args=training_args_fail,
    train_dataset=train_dataset_fail,
    eval_dataset=val_dataset_fail,
    compute_metrics=compute_metrics
)
trainer_fail.train()

# ─── 8. SAVE FINE-TUNED MODELS ─────────────────────────────────────────────
print("\n" + "="*70)
print("Saving Fine-tuned Models and Tokenizer...")
print("="*70)

os.makedirs("models", exist_ok=True)

# Save component classifier
model_component.save_pretrained('./models/tier3_component_classifier')
print("Saved component classifier to ./models/tier3_component_classifier")

# Save failure mode classifier  
model_failure.save_pretrained('./models/tier3_failure_classifier')
print("Saved failure mode classifier to ./models/tier3_failure_classifier")

# Save tokenizer once (shared by both)
tokenizer.save_pretrained('./models/tier3_tokenizer')
print("Saved tokenizer to ./models/tier3_tokenizer")

# Generate mappings
component_labels = list(df_train['component'].unique())
# Sort them by ID to ensure mapping is correct
comp_mapping = df_train[['component', 'component_id']].drop_duplicates().sort_values('component_id')
component_labels = list(comp_mapping['component'])
component2id = {c: int(i) for c, i in comp_mapping.values}
id2component = {int(i): c for c, i in comp_mapping.values}

fail_mapping = df_train[['failure_mode', 'failure_mode_id']].drop_duplicates().sort_values('failure_mode_id')
failure_labels = list(fail_mapping['failure_mode'])
failure2id = {f: int(i) for f, i in fail_mapping.values}
id2failure = {int(i): f for f, i in fail_mapping.values}

# Save label mappings
label_maps = {
    'component_labels': component_labels,
    'failure_labels': failure_labels,
    'component2id': component2id,
    'failure2id': failure2id,
    'id2component': id2component,
    'id2failure': id2failure
}
with open('./models/tier3_label_maps.json', 'w') as f:
    json.dump(label_maps, f, indent=2)
print("Saved label mappings to ./models/tier3_label_maps.json")

print("\nNLP Training Pipeline Finished Successfully!")
