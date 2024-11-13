from transformers import pipeline

# named entity recognition (ner)  find the entities in text such as person organization 
# person is PER  , organization is ORG and    location is LOC


ner = pipeline("ner", grouped_entities= True)

recognized =ner("My name is ibrahim and i study at FAST NUCES and i live in karachi")
 
print(recognized)

