
from transformers import pipeline



# question answering model 


question_answerer = pipeline("question-answering")
answered= question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

print(answered)
