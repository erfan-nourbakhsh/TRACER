
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BeliefStateEncoder(nn.Module):

    def __init__(self, model_name="roberta-base", freeze_layers=8,
                 output_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim

        if freeze_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            layers = self.encoder.encoder.layer
            for i in range(min(freeze_layers, len(layers))):
                for param in layers[i].parameters():
                    param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.Tanh(),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0]
        return self.projection(cls_output)

    @staticmethod
    def get_tokenizer(model_name="roberta-base"):
        return AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def belief_state_to_text(belief_state):
        if not belief_state:
            return "empty state"
        parts = []
        for slot, value in sorted(belief_state.items()):
            slot_text = slot.replace("-", " ").replace("_", " ")
            parts.append(f"{slot_text} is {value}")
        return ", ".join(parts)
