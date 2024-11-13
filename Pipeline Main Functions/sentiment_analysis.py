from transformers import pipeline


# sentiment analysis 

analysis = pipeline("sentiment-analysis")        # gives whther statement is positive and score 


analyze= analysis("I have tired of my daily hectic and boring routine")     #it should be negative 
print(analyze) 


