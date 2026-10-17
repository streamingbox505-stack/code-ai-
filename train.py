"""
LoRA/QLoRA fine-tuning kwa matumizi binafsi (kujifunza).
Dataset: MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x

Tumia:  python train.py --model Qwen/Qwen2.5-1.5B-Instruct --samples 500
"""
import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="modeli ya msingi (base)")
p.add_argument("--dataset", default="MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x")
p.add_argument("--samples", type=int, default=500, help="anza na kidogo ili kujaribu")
p.add_argument("--max_len", type=int, default=4096)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--out", default="out-lora")
p.add_argument("--no_4bit", action="store_true", help="zima QLoRA (inahitaji VRAM zaidi)")
p.add_argument("--dry_run", action="store_true", help="andaa data tu, bila kufundisha (hakuna GPU)")
p.add_argument("--cpu", action="store_true", help="tumia CPU tu (polepole sana, kwa majaribio madogo)")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def fix_tool_calls(messages):
    """Chat templates nyingi zinataka arguments kama dict, si string ya JSON."""
    for m in messages:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try:
                    fn["arguments"] = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    pass
    return messages


def to_text(ex):
    msgs = fix_tool_calls(ex["messages"])
    tools = ex.get("tools") or None
    try:
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    except Exception:
        text = tok.apply_chat_template(msgs, tokenize=False)
    return {"text": text}


ds = load_dataset(args.dataset, split="train")
ds = ds.select(range(min(args.samples, len(ds))))
ds = ds.map(to_text, remove_columns=ds.column_names)

if args.dry_run:
    print("DRY RUN: sampuli", len(ds), "| herufi za sampuli ya kwanza:", len(ds[0]["text"]))
    print(ds[0]["text"][:500])
    raise SystemExit(0)

quant = None
if not args.no_4bit and not args.cpu:
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=quant,
    torch_dtype=torch.float32 if args.cpu else torch.bfloat16,
    device_map=None if args.cpu else "auto",
)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

cfg = SFTConfig(
    output_dir=args.out,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=5,
    save_strategy="epoch",
    bf16=not args.cpu,
    use_cpu=args.cpu,
    gradient_checkpointing=True,
    max_length=args.max_len,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=ds,
    processing_class=tok,
    peft_config=lora,
)
trainer.train()
trainer.save_model(args.out)
print("Imekamilika. Adapter iko kwenye:", args.out)
