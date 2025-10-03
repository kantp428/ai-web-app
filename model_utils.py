import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.load_state_dict(torch.load("gpt2-finetuned-model-wikitext103.pt", map_location=device))
model.to(device)
model.eval()

def predict_with_model(text: str, top_k: int = 8):

    # input text -> predict tokens
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # logit ของ token ล่าสุด
    next_token_logits = logits[0, -1]

    # top-k token id
    top_token_ids = torch.topk(next_token_logits, top_k).indices.tolist()

    # decode กลับเป็น string
    top_tokens = [tokenizer.decode([tid]).strip() for tid in top_token_ids]
    return top_tokens