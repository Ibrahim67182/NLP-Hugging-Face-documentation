from transformers import pipeline

#zero shot classification




classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",   # o  a given context  and labels it will give predicted score for each label associated with context

    candidate_labels=["education", "politics", "business"],  # labels to get their score
)


