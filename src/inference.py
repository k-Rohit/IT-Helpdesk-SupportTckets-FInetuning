from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

MODEL_NAME = "kumarrohit1707/it-helpdesk-sft-adapter"
MAX_SEQ_LENGTH = 2048
SYSTEM_PROMPT = (
    "You are an IT Helpdesk Assistant. Answer user IT support questions "
    "clearly and professionally."
)

print("Loading IT Helpdesk Assistant model (SFT), please wait...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

tokenizer = get_chat_template(tokenizer, chat_template="chatml")
FastLanguageModel.for_inference(model)

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
model.generation_config.eos_token_id = im_end_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

print("Model loaded. Ready to answer questions.\n")


def ask_question(question: str, max_new_tokens: int = 250) -> str:
    """
    Send a question to the fine-tuned IT Helpdesk Assistant and return its answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=im_end_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    answer = tokenizer.decode(
        outputs[0][inputs.shape[1]:],
        skip_special_tokens=True,
    )
    return answer.strip()