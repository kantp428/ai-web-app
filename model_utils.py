import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ID บน Hugging Face Hub
MODEL_ID = "mrktp/text-mini-gpt2-finetuned"

# โหลด model + tokenizer ทันทีที่เริ่ม run
print("🔄 Loading model... please wait.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
model.eval()
print("✅ Model loaded successfully.")

def predict_with_model(text: str, top_k: int = 8):
    """รับ input text แล้วคืน top-k predicted next tokens"""

    # tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # logit ของ token ล่าสุด
    next_token_logits = logits[0, -1]

    # เลือก top-k token id
    top_token_ids = torch.topk(next_token_logits, top_k).indices.tolist()

    # decode กลับเป็น string และ strip whitespace
    top_tokens = [
        tokenizer.decode([tid], skip_special_tokens=True).strip()
        for tid in top_token_ids
        if tokenizer.decode([tid], skip_special_tokens=True).strip()  # กัน token ว่าง
    ]

    return top_tokens
