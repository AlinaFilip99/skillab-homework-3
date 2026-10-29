"""Intent Router - Clasifică întrebări pentru direcționare."""
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

Intent = Literal["rag", "sql", "chat"]


class IntentClassifier:
    def __init__(
        self,
        model_path: str | Path = "models/intent_classifier",
        labels: list[str] | None = None,
        device: str | None = None,
    ):
        self.model_path = Path(model_path)
        self.labels = labels or ["rag", "sql"]
        self._tokenizer = None
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        logger.info(f"Loading intent classifier from {self.model_path}")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self._model.eval()

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        self._model.to(self._device)
        logger.info(f"Intent classifier loaded on {self._device}")

    def classify(self, query: str) -> tuple[Intent, float]:
        import torch

        self._load_model()

        inputs = self._tokenizer(
            query, return_tensors="pt", truncation=True, max_length=128
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        idx = probs.argmax().item()
        intent = self.labels[idx]
        confidence = probs[idx].item()

        logger.info(f"[CLASSIFIER] '{query[:50]}...' → {intent} ({confidence:.1%})")
        return intent, confidence


_classifier: IntentClassifier | None = None


def get_classifier(model_path: str | None = None) -> IntentClassifier:
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier(model_path=model_path or "models/intent_classifier")
    return _classifier


def route(query: str, intent: Intent | None = None) -> Intent:
    if intent is not None:
        logger.info(f"[ROUTER] Manual intent: {intent}")
        return intent

    # Auto-detect only between rag and sql
    classifier = get_classifier()
    detected_intent, _ = classifier.classify(query)
    return detected_intent
