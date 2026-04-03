from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
from rapidfuzz import process, fuzz
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    T5Tokenizer,
    T5ForConditionalGeneration
)

# Load models once (IMPORTANT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sentiment_model = DistilBertForSequenceClassification.from_pretrained("Jay2465/distilbert-sentiment").to(device)

sentiment_tokenizer = DistilBertTokenizerFast.from_pretrained("Jay2465/distilbert-sentiment")

t5_model = T5ForConditionalGeneration.from_pretrained("Jay2465/t5-absa").to(device)

t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

app = FastAPI()

# Request format
class TextInput(BaseModel):
    text: str



food_list = ["idli","dosa","puri","poori","puri masala","pongal","upma","rava idli", "rava khichdi","semiya kichadi","idiyappam","vada curry", "poha","poha bhujiya","pav bhaji", "bread","toasted bread","wheat bread","butter","jam","sandwich", "white rice","veg pulao","green peas pulao","ghee rice", "jeera rice","schezwan fried rice","veg fried rice","egg fried rice", "chapathi","phulka", "dal fry","methi dal","jeera dal","ahar dal","ahar dal tadka", "panchamel dal","andhara pappu", "sambar","chettinad sambar","uduppi sambar", "arachivitta sambar","rasam","milagu rasam", "paruppu rasam","moong dal rasam","pineapple rasam", "uduppi tomato rasam","Biryani","Chicken Biryani", "Veg biryani","Paneer Butter Masala", "mixed veg sabzi","mixed veg pulao","veg sabzi", "potato pattani masala","cabbage poriyal", "cabbage aloo poriyal","beans poriyal", "carrot beans poriyal","raw banana poriyal", "keerai poriyal","tendli poriyal", "dry cabbage poriyal","mixed vegetable poriyal", "kadai paneer","paneer tikka masala","palak paneer", "mughalai chicken","chicken handi masala", "chicken chettinad","tomato egg curry", "boiled egg","egg bhurji","egg curry","egg fried rice", "rajma curry","white chenna masala","soya chunk gravy", "boiled chana chat", "bhel puri","pani puri","sweet potato","corn","potato chips", "vaazhai thandu soup","spicy mushroom soup", "clear soup","sweet corn soup","hot and sour soup", "chicken soup","veg soup", "salad","cucumber salad","carrot salad", "beetroot salad","tomato salad", "ice cream","sweet","fruit salad", "banana","papaya","watermelon","apple", "black grapes","pineapple","muskmelon", "tea","coffee","milk","buttermilk", "sweet lassi","badam milk", "lemon mint juice","mosambi juice", "apple juice","grape juice", "pineapple juice","watermelon juice", "cornflakes","chocos", "paruppu podi","gingelly oil","thovaiyal", "pickle","papad","fryums","podi oil", "boondi raita"]  # your full list
normalized_food_list = [f.lower().strip() for f in food_list]

def match_food(predicted_food):
    if not predicted_food:
        return "general"

    query = predicted_food.lower().strip()

    match = process.extractOne(
        query,
        normalized_food_list,
        scorer=fuzz.token_sort_ratio,  # better for multi-word items
        score_cutoff=80
    )

    if match:
        matched_text = match[0]
        return matched_text.title()
    else:
        return "general"
    

def split_sentences(text):
    # Split on punctuation + contrast words
    parts = re.split(r'\.|\bbut\b|\bhowever\b|\balthough\b', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]

# Your function (simplified copy)
def predict_pipeline(sentence):
    sentences = split_sentences(sentence)
    results = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        input_text = f"Extract food, aspect, opinion from: {sent}"
        inputs = t5_tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            outputs = t5_model.generate(**inputs, max_length=50)

        extraction = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = {"food": "", "aspect": "", "opinion": ""}

        try:
            parts = extraction.split(",")
            for part in parts:
                key, value = part.split(":")
                result[key.strip()] = value.strip()
        except:
            pass
        
        result["food"] = match_food(result["food"])
        # sentiment
        inputs = sentiment_tokenizer(sent, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            logits = sentiment_model(**inputs).logits
            sentiment = torch.argmax(logits).item()

        result["sentiment"] = "positive" if sentiment == 1 else "negative"

        results.append(result)

    return results


# API endpoint
@app.post("/predict")
def predict(data: TextInput):
    result = predict_pipeline(data.text)
    return result