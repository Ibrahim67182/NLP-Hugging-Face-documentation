from transformers import pipeline


# mask filling model 

unmasker = pipeline("fill-mask")
unmasked= unmasker("The AI course will teach you all about <mask> skills.", top_k=2)   # it fill two words in mask 

print(unmasked)

