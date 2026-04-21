
import argparse
import json
import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracer.recovery.t5_recovery import (
    build_t5_input,
    belief_state_to_str,
    format_context_turns,
    extract_confirmation_utterances_from_mwoz,
)
from tracer.data.unified import UnifiedDialogue, UnifiedTurn


def load_config(config_path: str):
    import omegaconf
    return omegaconf.OmegaConf.load(config_path)


def load_unified_from_cache(cache_dir: str, split: str):
    path = os.path.join(cache_dir, f"unified_{split}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f)
    dialogues = []
    for d in data:
        if d.get("dataset") != "mwoz":
            continue
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
    return dialogues


class T5RecoveryDataset(Dataset):

    def __init__(self, pairs, tokenizer, max_input_len=256, max_output_len=64):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_str, target_str = self.pairs[idx]
        enc = self.tokenizer(
            input_str,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            target_str,
            max_length=self.max_output_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec_labels = dec["input_ids"].squeeze(0).clone()
        dec_labels[dec_labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": dec_labels,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.data.get("cache_dir", "cache")
    out_dir = args.output_dir or os.path.join(cfg.output.get("checkpoint_dir", "outputs"), "t5_recovery")
    t5_model_name = cfg.recovery.get("t5_model", "t5-small")
    os.makedirs(out_dir, exist_ok=True)

    train_dialogues = load_unified_from_cache(cache_dir, "train")
    if not train_dialogues:
        print("No MultiWOZ train data in cache. Run preprocess_all.py first.")
        return

    pairs = extract_confirmation_utterances_from_mwoz(train_dialogues)
    if len(pairs) < 100:
        print(f"Only {len(pairs)} confirmation pairs extracted. Consider lowering thresholds or adding more data.")
    print(f"Extracted {len(pairs)} (input, target) pairs for T5 recovery training.")

    tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    dataset = T5RecoveryDataset(pairs, tokenizer, max_input_len=256, max_output_len=64)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1} avg_loss={avg_loss:.4f}")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"T5 recovery model saved to {out_dir}")


if __name__ == "__main__":
    main()
