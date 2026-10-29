"""
Fine-tune BERT pentru intent classification.

Model: dumitrescustefan/bert-base-romanian-cased-v1

Usage:
    python -m src.agents.train_classifier
    python -m src.agents.train_classifier --output models/my_classifier
"""
import argparse
import logging
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_MODEL = "dumitrescustefan/bert-base-romanian-cased-v1"
DEFAULT_OUTPUT = "models/intent_classifier"

LABELS = ["rag", "sql"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# Training data - exemple pentru SEAP
TRAIN_DATA = [
    # === RAG (documente, proceduri, explicații) ===
    ("Ce este o achiziție directă?", "rag"),
    ("Care sunt pașii pentru o licitație deschisă?", "rag"),
    ("Cum funcționează procedura de negociere?", "rag"),
    ("Ce documente trebuie pentru DUAE?", "rag"),
    ("Explică-mi ce înseamnă CPV", "rag"),
    ("Care sunt pragurile pentru achiziții publice?", "rag"),
    ("Ce condiții trebuie îndeplinite pentru participare?", "rag"),
    ("Unde găsesc regulamentul de achiziții?", "rag"),
    ("Ce spune legea despre subcontractare?", "rag"),
    ("Care sunt criteriile de calificare?", "rag"),
    ("Cum se calculează termenele de depunere?", "rag"),
    ("Ce înseamnă fonduri europene în achiziții?", "rag"),
    ("Cum se contestă o licitație?", "rag"),
    ("Ce este garanția de participare?", "rag"),
    ("Explică procedura simplificată", "rag"),
    ("Ce documente trebuie pentru ofertă?", "rag"),
    ("Cum se evaluează ofertele?", "rag"),
    ("Ce este acordul cadru?", "rag"),
    ("Găsește informații despre SICAP", "rag"),
    ("Ce trebuie să conțină caietul de sarcini?", "rag"),

    # === SQL (date, statistici, căutări în tabele) ===
    ("Câte achiziții au fost în 2024?", "sql"),
    ("Care sunt cele mai mari contracte?", "sql"),
    ("Arată firmele cu cele mai multe contracte", "sql"),
    ("Listează achizițiile din București", "sql"),
    ("Top 10 furnizori după valoare", "sql"),
    ("Care e valoarea totală a achizițiilor din martie?", "sql"),
    ("Câte licitații a câștigat firma X?", "sql"),
    ("Găsește contractele cu CPV 45000000", "sql"),
    ("Care autorități au cheltuit cel mai mult?", "sql"),
    ("Medie valoare contracte pe lună", "sql"),
    ("Listează achizițiile peste 100000 euro", "sql"),
    ("Câți furnizori unici avem?", "sql"),
    ("Care sunt achizițiile din sectorul IT?", "sql"),
    ("Sumă totală contracte anul trecut", "sql"),
    ("Arată contractele semnate în Q1", "sql"),
    ("Număr achiziții per județ", "sql"),
    ("Care sunt procedurile de urgență?", "sql"),
    ("Listează firmele cu CUI-ul 12345678", "sql"),
    ("Top autorități contractante", "sql"),
    ("Valoare medie achiziții directe", "sql"),
    ("Găsește toate contractele cu valoare între 50000 și 100000", "sql"),
    ("Care sunt licitațiile din ultima săptămână?", "sql"),
    ("Afișează sumele pe categorii CPV", "sql"),
    ("Câte contracte are autoritatea X?", "sql"),
]


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for intent classification")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Fine-tuning {BASE_MODEL}")
    logger.info(f"Data: {len(TRAIN_DATA)} examples, {len(LABELS)} classes")
    logger.info(f"Output: {output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Dataset
    dataset = Dataset.from_dict({
        "text": [t for t, _ in TRAIN_DATA],
        "label": [LABEL2ID[label] for _, label in TRAIN_DATA],
    })

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    logger.info(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train
    logger.info("Training...")
    trainer.train()

    # Save
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

    # Test
    logger.info("\nTesting on new queries:")
    test_queries = [
        "Ce este o licitație?",
        "Câte contracte avem?",
        "Explică procedura",
        "Top furnizori",
    ]

    model.eval()
    for q in test_queries:
        inputs = tokenizer(q, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        idx = probs.argmax().item()
        logger.info(f"  '{q}' → {LABELS[idx]} ({probs[idx]:.0%})")


if __name__ == "__main__":
    main()
