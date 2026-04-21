
import torch
import torch.nn as nn

from .temporal_transformer import TemporalTransformer
from .belief_encoder import BeliefStateEncoder


class TRACERPredictor(nn.Module):

    def __init__(self, stream_a_config, stream_b_config, fusion_config):
        super().__init__()

        self.stream_a = TemporalTransformer(
            input_dim=stream_a_config.get("input_dim", 5),
            d_model=stream_a_config.get("d_model", 64),
            nhead=stream_a_config.get("nhead", 4),
            num_layers=stream_a_config.get("num_layers", 3),
            dropout=stream_a_config.get("dropout", 0.1),
        )

        self.stream_b = BeliefStateEncoder(
            model_name=stream_b_config.get("encoder_name", "roberta-base"),
            freeze_layers=stream_b_config.get("freeze_layers", 8),
            output_dim=stream_b_config.get("output_dim", 256),
        )

        d_model = stream_a_config.get("d_model", 64)
        output_dim = stream_b_config.get("output_dim", 256)
        hidden_dim = fusion_config.get("hidden_dim", 256)
        fusion_dropout = fusion_config.get("dropout", 0.2)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory_features, traj_mask, input_ids, attention_mask):
        h_a = self.stream_a(trajectory_features, traj_mask)
        h_b = self.stream_b(input_ids, attention_mask)
        fused = torch.cat([h_a, h_b], dim=-1)
        logits = self.fusion_mlp(fused).squeeze(-1)
        return logits

    def get_stream_a_params(self):
        return list(self.stream_a.parameters()) + list(self.fusion_mlp.parameters())

    def get_stream_b_params(self):
        return list(self.stream_b.parameters())


class FeaturesOnlyPredictor(nn.Module):

    def __init__(self, stream_a_config, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.stream_a = TemporalTransformer(
            input_dim=stream_a_config.get("input_dim", 5),
            d_model=stream_a_config.get("d_model", 64),
            nhead=stream_a_config.get("nhead", 4),
            num_layers=stream_a_config.get("num_layers", 3),
            dropout=stream_a_config.get("dropout", 0.1),
        )
        d_model = stream_a_config.get("d_model", 64)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory_features, traj_mask, **kwargs):
        h = self.stream_a(trajectory_features, traj_mask)
        return self.classifier(h).squeeze(-1)


class TextOnlyPredictor(nn.Module):

    def __init__(self, stream_b_config, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.stream_b = BeliefStateEncoder(
            model_name=stream_b_config.get("encoder_name", "roberta-base"),
            freeze_layers=stream_b_config.get("freeze_layers", 8),
            output_dim=stream_b_config.get("output_dim", 256),
        )
        output_dim = stream_b_config.get("output_dim", 256)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        h = self.stream_b(input_ids, attention_mask)
        return self.classifier(h).squeeze(-1)

    def get_stream_b_params(self):
        return list(self.stream_b.parameters())

    def get_stream_a_params(self):
        return list(self.classifier.parameters())


class EarlyPredictionLoss(nn.Module):

    def __init__(self, early_lambda=0.3, pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        else:
            self.pos_weight = None
        self.early_lambda = early_lambda

    def forward(self, logits, labels, turn_positions, dialogue_lengths):
        pos_weight = self.pos_weight
        if pos_weight is not None and pos_weight.device != logits.device:
            pos_weight = pos_weight.to(logits.device)

        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=pos_weight,
            reduction="none",
        )

        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = (predictions == labels).float()
        failure_correct = correct * labels

        weights = (dialogue_lengths.float() - turn_positions.float()) / \
                  dialogue_lengths.float().clamp(min=1)

        early_reward = failure_correct * weights

        loss = bce_loss.mean() - self.early_lambda * early_reward.mean()
        return loss
