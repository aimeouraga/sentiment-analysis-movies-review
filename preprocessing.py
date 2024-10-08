import numpy as np
import torch
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os

load_dotenv()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

with open(os.getenv("TOKENIZER_PATH"), 'r') as f:
    tokenizer = json.load(f)

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Lowercase the text
    text = ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english')])  # Remove stopwords
    return text

def text_to_sequence(text):
    return [tokenizer.get(word, 0) for word in word_tokenize(text)] 

def pad_sequence(sequence, maxlen= int(os.getenv("MAX_LENGTH"))):
   
    return np.array(np.pad(sequence, (0, max(0, maxlen - len(sequence))), 'constant')[:maxlen])


def preprocess_text(text):
    text = clean_text(text)
    sequence = text_to_sequence(text)
    padded_seq = pad_sequence(sequence)
    padded_seq = torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0)
    return padded_seq

