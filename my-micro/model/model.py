from transformers import AutoTokenizer, BartForConditionalGeneration
from . import initiale


class info:
    state:int

info.state=0

(model,tokenizer) = initiale.downloadModel()


def summarizer(test:str):
    inputs = tokenizer([test], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=0, max_length=20)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]