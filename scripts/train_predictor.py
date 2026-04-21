
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.data.unified import (
    UnifiedDialogue,
    UnifiedTurn,
    PrefixDialogueDataset,
    collate_dialogue_batch,
)
from tracer.models.dual_stream import TRACERPredictor, EarlyPredictionLoss
from tracer.models.belief_encoder import BeliefStateEncoder
from tracer.evaluation.metrics import compute_auc_roc, compute_early_detection_score
from tracer.utils.training_utils import (
    set_seed,
    get_linear_warmup_cosine_scheduler,
    save_checkpoint,
    EarlyStopping,
    compute_pos_weight,
    setup_logging,
)


def load_config(config_path: str):
    import omegaconf
    return omegaconf.OmegaConf.load(config_path)


def load_unified_and_features(cache_dir: str, split: str):
    path_dial = os.path.join(cache_dir, f"unified_{split}.json")
    path_feat = os.path.join(cache_dir, f"features_{split}.json")
    if not os.path.exists(path_dial) or not os.path.exists(path_feat):
        return [], {}
    with open(path_dial, "r") as f:
        data = json.load(f)
    dialogues = []
    for d in data:
        turns = [
            UnifiedTurn(
                turn_idx=t["turn_idx"],
                user_utterance=t["user_utterance"],
                system_utterance=t["system_utterance"],
                belief_state=t["belief_state"],
                turn_delta=t["turn_delta"],
                domain=t["domain"],
            )
            for t in d["turns"]
        ]
        dialogues.append(UnifiedDialogue(
            dialogue_id=d["dialogue_id"],
            dataset=d["dataset"],
            domains=d["domains"],
            turns=turns,
            success=d["success"],
            metadata=d.get("metadata", {}),
        ))
    with open(path_feat, "r") as f:
        features_dict = json.load(f)
    for k in features_dict:
        features_dict[k] = [np.array(x, dtype=np.float32) for x in features_dict[k]]
    return dialogues, features_dict


def build_dialogue_prediction_sequences(predictions, labels, dialogue_ids, turn_positions):
    dialogue_predictions = {}
    dialogue_labels = {}

    for pred, label, dial_id, turn_idx in zip(predictions, labels, dialogue_ids, turn_positions):
        if dial_id not in dialogue_predictions:
            dialogue_predictions[dial_id] = []
        dialogue_predictions[dial_id].append((int(turn_idx), float(pred)))
        dialogue_labels[dial_id] = bool(label == 0)

    sorted_predictions = {}
    for dial_id, seq in dialogue_predictions.items():
        seq = sorted(seq, key=lambda x: x[0])
        sorted_predictions[dial_id] = [pred for _, pred in seq]

    return sorted_predictions, dialogue_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--dataset_filter", type=str, default="mwoz", help="mwoz, sgd, abcd, or all")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.training.get("seed", 42))

    cache_dir = cfg.data.get("cache_dir", "cache")
    out_dir = cfg.output.get("checkpoint_dir", "outputs")
    log_dir = cfg.output.get("log_dir", os.path.join(out_dir, "logs"))
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logging(log_dir, "train_predictor")

    train_dialogues, train_features = load_unified_and_features(cache_dir, "train")
    dev_dialogues, dev_features = load_unified_and_features(cache_dir, "dev")

    if args.dataset_filter != "all":
        train_dialogues = [d for d in train_dialogues if d.dataset == args.dataset_filter]
        dev_dialogues = [d for d in dev_dialogues if d.dataset == args.dataset_filter]
        train_features = {d.dialogue_id: train_features[d.dialogue_id] for d in train_dialogues if d.dialogue_id in train_features}
        dev_features = {d.dialogue_id: dev_features[d.dialogue_id] for d in dev_dialogues if d.dialogue_id in dev_features}

    if not train_dialogues or not train_features:
        logger.error("No training data. Run preprocess_all.py and compute_features.py first.")
        return

    tokenizer = BeliefStateEncoder.get_tokenizer(cfg.model.stream_b.get("encoder_name", "roberta-base"))
    max_length = cfg.model.stream_b.get("max_length", 512)

    train_dataset = PrefixDialogueDataset(
        train_dialogues,
        train_features,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dev_dataset = PrefixDialogueDataset(
        dev_dialogues,
        dev_features,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.get("batch_size", 32),
        shuffle=True,
        collate_fn=collate_dialogue_batch,
        num_workers=cfg.training.get("num_workers", 0),
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.training.get("batch_size", 32),
        shuffle=False,
        collate_fn=collate_dialogue_batch,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TRACERPredictor(
        stream_a_config=dict(cfg.model.stream_a),
        stream_b_config=dict(cfg.model.stream_b),
        fusion_config=dict(cfg.model.fusion),
    )
    model.to(device)

    all_labels = np.array([sample["label"].item() for sample in train_dataset])
    pos_weight = compute_pos_weight(all_labels)
    criterion = EarlyPredictionLoss(
        early_lambda=cfg.training.get("early_pred_lambda", 0.3),
        pos_weight=pos_weight,
    )

    stream_a_params = model.get_stream_a_params()
    stream_b_params = model.get_stream_b_params()
    optimizer = torch.optim.AdamW(
        [
            {"params": stream_a_params, "lr": cfg.training.lr},
            {"params": stream_b_params, "lr": cfg.training.lr * cfg.training.get("encoder_lr_factor", 0.1)},
        ]
    )

    num_epochs = cfg.training.get("epochs", 30)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * cfg.training.get("warmup_ratio", 0.1))
    scheduler = get_linear_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    early_stopping = EarlyStopping(
        patience=cfg.training.get("early_stop_patience", 5),
        mode="max",
    )

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from %s at epoch %s", args.resume, start_epoch)

    best_auc = 0.0
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            features = batch["features"].to(device)
            traj_mask = batch["traj_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).float()
            turn_positions = batch["turn_position"].to(device)
            dialogue_lengths = batch["dialogue_length"].to(device)

            optimizer.zero_grad()
            logits = model(features, traj_mask, input_ids, attention_mask)
            loss = criterion(logits, labels, turn_positions, dialogue_lengths)
            loss.backward()
            if cfg.training.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        model.eval()
        dev_preds, dev_labels = [], []
        dev_dialogue_ids, dev_turn_positions = [], []
        with torch.no_grad():
            for batch in dev_loader:
                features = batch["features"].to(device)
                traj_mask = batch["traj_mask"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(features, traj_mask, input_ids, attention_mask)
                probs = torch.sigmoid(logits).cpu().numpy()
                labs = batch["labels"].numpy()
                turn_positions = batch["turn_position"].numpy()
                dev_preds.extend(probs)
                dev_labels.extend(labs)
                dev_dialogue_ids.extend(batch.get("dialogue_id", []))
                dev_turn_positions.extend(turn_positions.tolist())

        auc = compute_auc_roc(dev_preds, dev_labels)
        dial_preds, dial_labels = build_dialogue_prediction_sequences(
            dev_preds, dev_labels, dev_dialogue_ids, dev_turn_positions
        )
        eds_result = compute_early_detection_score(dial_preds, dial_labels, threshold=0.5)
        logger.info(
            "Epoch %d train_loss=%.4f dev_auc=%.4f dev_eds=%.4f mean_detect=%.2f sanity=%s",
            epoch,
            train_loss,
            auc,
            eds_result["eds"],
            eds_result["mean_detection_turn"],
            eds_result["sanity_check_passed"],
        )

        if auc > best_auc:
            best_auc = auc
            ckpt_path = os.path.join(out_dir, "best_predictor.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, {"auc": auc, "eds": eds_result["eds"]}, ckpt_path)
            logger.info("Saved best checkpoint to %s", ckpt_path)

        if early_stopping(auc):
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Training finished. Best dev AUC: %.4f", best_auc)


if __name__ == "__main__":
    main()
