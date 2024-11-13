from transformers import pipeline

#zero shot classification




classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",   //context 

    candidate_labels=["education", "politics", "business"],   labels to get their score
)


