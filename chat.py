import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

with open('intents.json','r') as f:
    intents=json.load(f)

FILE="data.pth"
data=torch.load(FILE)
input_size=data["input_size"]
hidden_size=data["hidden_size"]
output_size=data["output_size"]
all_words=data["all_words"]
tags=data["tags"]
model_state=data["model_state"]

model=NeuralNet(input_size,hidden_size,output_size)
model.load_state_dict(model_state)
model.eval()

print("Let's chat!! type quit if you have had enough of me :(")
while True:
    sentence=input("You: ")
    if sentence=="quit":
        break
    sentence=tokenize(sentence)
    X=bag_of_words(sentence, all_words)
    X=X.reshape(1,X.shape[0])
    X=torch.from_numpy(X)

    output=model(X)
    _,predicted=torch.max(output, dim=1)
    tag= tags[predicted.item()]
    probs=torch.softmax(output,dim=1)
    prob=probs[0][predicted.item()]
    
    if prob.item()>0.50:
        for intent in intents["intents"]:
            if tag==intent["tag"]:
                print("Chutki: ", random.choice(intent["responses"]))
    else:
        print("Chutki: ",random.choice(["I understand you, but why don't you try talking to your amma/bapu as well? I'm sure they can help you","Okay then, have a laddu","Sorry, I didn't understand you.", "Please go on.", "Why don't you try a laddu?", "Please don't hesitate to talk to me."]))
            
