from transformers import AutoTokenizer          # hugging face  library class "auto tokenizer" automatically select suitable tokenizer according to the given model checkpoint passed


# step 1 preprocessing using tokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"        # here we are using sentiment analysis and this checkpoint represents it

tokenizer = AutoTokenizer.from_pretrained(checkpoint)                 #storing tokenizer in varibale by using method from_pretrained() which fetched the tokenizer 



#step 2 passing raw text in tokenizer to make it understand by ML Model 
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")

print(inputs)


#step 3 now finding suitable ML model to given data in tokenizer

from transformers import TFAutoModel                               #tensor flow model class to find model 

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = TFAutoModel.from_pretrained(checkpoint)


outputs = model(inputs)                    #passed the numeric input achieve in previous step into model 

print(outputs.last_hidden_state.shape)         # printing high dimensional vector 


# can also used sequence classification model to classify sentence as positive or negative in this case TFAutoModel cant used

from transformers import TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs = model(inputs)

print(outputs.logits.shape)



#step 4 now convert these logits scores into predicted probabilites softmax fucntion in tensor flow

import tensorflow as tf

predictions = tf.math.softmax(outputs.logits, axis=-1)

print(predictions)               # now this will print probability socres of each sentences 


model.config.id2label             # this is now used to identify labels positive or negative labels of each sentence 








