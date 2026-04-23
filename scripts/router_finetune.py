"""
라우터 파인튜닝
  베이스 모델 : klue/roberta-base
  태스크     : 4-class 텍스트 분류
  출력       : ./router_model/
"""

import csv
import json
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
BASE_MODEL  = "klue/roberta-base"
DATA_CSV    = "./router_train.csv"
OUTPUT_DIR  = "./router_model"
MAX_LEN     = 64    # 라우터 입력은 짧으므로 64 충분
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 2e-5

LABEL2ID = {
    "recommendation": 0,
    "product_info":   1,
    "special_terms":  2,
    "general_qa":     3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
def load_data(path: str):
    texts, labels = [], []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            text  = row["text"].strip()
            label = row["label"].strip()
            if text and label in LABEL2ID:
                texts.append(text)
                labels.append(LABEL2ID[label])
    return texts, labels


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class RouterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ─────────────────────────────────────────
# Weighted Loss (클래스 불균형 처리)
# ─────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────
# 평가 지표
# ─────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc   = (preds == labels).mean()

    # 클래스별 정확도
    per_class = {}
    for id_, name in ID2LABEL.items():
        mask = labels == id_
        if mask.sum() > 0:
            per_class[name] = round((preds[mask] == labels[mask]).mean(), 4)

    return {"accuracy": round(float(acc), 4), **per_class}


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    print("=" * 50)
    print("  라우터 파인튜닝 시작")
    print("=" * 50)

    # 1. 데이터 로드
    texts, labels = load_data(DATA_CSV)
    print(f"\n총 데이터: {len(texts)}개")
    counts = Counter(labels)
    for id_, name in ID2LABEL.items():
        print(f"  {name:20s}: {counts[id_]}개")

    # 2. Train / Eval 분리 (8:2)
    tr_texts, ev_texts, tr_labels, ev_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\nTrain: {len(tr_texts)}개 / Eval: {len(ev_texts)}개")

    # 3. 토크나이저 & 모델 로드
    print(f"\n[모델 로딩] {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # 4. Dataset 생성
    train_dataset = RouterDataset(tr_texts, tr_labels, tokenizer)
    eval_dataset  = RouterDataset(ev_texts, ev_labels, tokenizer)

    # 5. 클래스 가중치 계산 (샘플 수 역수)
    total = len(tr_labels)
    class_weights = torch.tensor([
        total / (len(LABEL2ID) * counts[i]) for i in range(len(LABEL2ID))
    ], dtype=torch.float)
    print(f"\n클래스 가중치: { {ID2LABEL[i]: round(w.item(), 2) for i, w in enumerate(class_weights)} }")

    # 6. 학습 설정
    training_args = TrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        learning_rate               = LR,
        warmup_ratio                = 0.1,
        weight_decay                = 0.01,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "accuracy",
        logging_steps               = 10,
        fp16                        = torch.cuda.is_available(),
        report_to                   = "none",
    )

    # 7. Trainer
    trainer = WeightedTrainer(
        class_weights = class_weights,
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = eval_dataset,
        compute_metrics = compute_metrics,
    )

    # 8. 학습
    print("\n[학습 시작]")
    trainer.train()

    # 9. 최종 평가
    print("\n[최종 평가]")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k}: {v}")

    # 10. 모델 저장
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # label 매핑 저장 (router.py에서 사용)
    with open(f"{OUTPUT_DIR}/label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장 완료 → {OUTPUT_DIR}/")
    print("=" * 50)


if __name__ == "__main__":
    main()
