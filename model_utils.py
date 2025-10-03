import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# โหลด tokenizer + model จาก Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("mrktp/text-mini-gpt2-finetuned")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("mrktp/text-mini-gpt2-finetuned")
model.to(device)
model.eval()

def predict_with_model(text: str, top_k: int = 8):
    # tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # logit ของ token ล่าสุด
    next_token_logits = logits[0, -1]

    # top-k token id
    top_token_ids = torch.topk(next_token_logits, top_k).indices.tolist()

    # decode กลับเป็น string
    top_tokens = [tokenizer.decode([tid], skip_special_tokens=True).strip() for tid in top_token_ids]
    return top_tokens
