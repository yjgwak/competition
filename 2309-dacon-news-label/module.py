import lightning as L
import torch
from torch.utils.data import Dataset
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


class DeBERTaClassifier(L.LightningModule):
    def __init__(self, model_name, num_labels, learning_rate, num_epochs, num_batch, freeze_backbone=False):
        super(DeBERTaClassifier, self).__init__()
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, from_tf=True
            )
        self.model.requires_grad_(not freeze_backbone)
        self.classifier = torch.nn.Linear(len(self.model.config.id2label), num_labels)
        self.loss = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_batch = num_batch

    def forward(self, input_ids, attention_mask):
        return self.classifier(
            self.model(input_ids, attention_mask=attention_mask).logits
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["input_ids"]
        mask = batch["attention_mask"]
        labels = batch["labels"]
        outputs = self(inputs, mask)
        loss = self.loss(outputs, labels)
        return {
            "loss": loss,
            "log": {
                "train_loss": loss,
            },
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        inputs = batch["input_ids"]
        mask = batch["attention_mask"]
        outputs = self(inputs, mask)
        return torch.nn.functional.softmax(outputs)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_batch * self.num_epochs,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row["text"]
        category = row.get("category", None)

        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        for k, v in encoding.items():
            encoding[k] = torch.squeeze(v)
        if category is not None:
            encoding["labels"] = torch.tensor(category).long()
        else:
            encoding["labels"] = torch.tensor(-1).long()
        return encoding
