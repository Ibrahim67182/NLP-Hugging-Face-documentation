# encoding process in tokenizers from raw text to input ids 


from transformers import AutoTokenizer

name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(name)

text = "I’ve been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(text)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)


# decoding the string from input ids  reverse process

result  = tokenizer.decode([2026, 2171, 2003, 13477, 12022, 14326, 1998, 1045, 2572, 19875, 3076, 999])

print(result)


