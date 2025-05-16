import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Convert these dates to YYYY/MM/DD format:\n12/31/2021\n02-01-22"
messages = [
    {"role": "system", "content": "You are a helpful AI assistant developed by Kakao."},
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cpu")

_ = model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=72,
        do_sample=False,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))

# Output:
# Sure! Here are the given dates converted to the `YYYY/MM/DD` format:

# 1. **12/31/2021**
#    - **YYYY/MM/DD:** 2021/12/31

# 2. **02-01-22**
#    - **YYYY/MM/DD:** 2022/02/01

# So, the converted dates are ...
