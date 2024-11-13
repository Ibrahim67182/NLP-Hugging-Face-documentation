


from transformers import pipeline

# text generation 


generator = pipeline("text-generation", model="distilgpt2")
genrated= generator(
    "once upon a time",
    max_length=15,
    num_return_sequences=2,truncation= True                 #it will generate  two sequences or sentences of len 15 of a given context
)

print(genrated)
