

from transformers import pipeline

# translation text 
translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-ru")      # using nlp model of hugging face translator eng to russian

translated= translation("My name is ibrahim")

print(translated)
