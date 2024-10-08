import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from dotenv import load_dotenv
import os

load_dotenv()
# Define your model architecture (ensure it matches the training architecture)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, embedding_matrix=None):
        super(SentimentLSTM, self).__init__()

        # Load pre-trained embeddings
        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Adding an extra dense layer
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep
        x = F.dropout(F.relu(self.fc1(lstm_out)), p=0.3)  # Apply dropout with probability 0.3
        x = self.fc2(x)
        return self.sigmoid(x)


def load_model(): 
    # Load the tokenizer dictionary
    with open(os.getenv("TOKENIZER_PATH"), 'r') as f:
        tokenizer = json.load(f)

    # Load the saved embedding matrix
    embedding_matrix = np.load(os.getenv("EMBEDDING_PATH"))
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

    # Initialize the model with the embedding matrix
    vocab_size = len(tokenizer) + 1
    model = SentimentLSTM(vocab_size, embedding_dim=embedding_matrix.shape[1], embedding_matrix=embedding_matrix)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(os.getenv("MODEL_PATH"), map_location=torch.device('cpu')))
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    return model, device

def predict_sentiment(model, device, input_tensor):
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probability = output.item()  # Get the probability score from the model
        
        # Determine the sentiment label and confidence score
        if probability >= 0.5:
            sentiment_label = "Negative"
            confidence_score = probability  # Confidence in it being Positive
        else:
            sentiment_label = "Positive"
            confidence_score = 1 - probability  # Confidence in it being Negative

    return sentiment_label, confidence_score


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import json
# from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # Azure Blob Storage connection
# AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
# MODEL_URL = os.getenv("MODEL_URL")
# TOKENIZER_URL = os.getenv("TOKENIZER_URL")
# EMBEDDING_URL = os.getenv("EMBEDDING_URL")

# blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# def download_blob_to_file(blob_url, local_filename):
#     blob_client = blob_service_client.get_blob_client(container="project2", blob=blob_url.split('/')[-1])
#     with open(local_filename, "wb") as download_file:
#         download_file.write(blob_client.download_blob().readall())

# # Define your model architecture (ensure it matches the training architecture)
# class SentimentLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, embedding_matrix=None):
#         super(SentimentLSTM, self).__init__()

#         # Load pre-trained embeddings
#         if embedding_matrix is not None:
#             self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
#         else:
#             self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=True)
#         self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Adding an extra dense layer
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep
#         x = F.dropout(F.relu(self.fc1(lstm_out)), p=0.3)  # Apply dropout with probability 0.3
#         x = self.fc2(x)
#         return self.sigmoid(x)

# def load_model():
#     # Download the tokenizer, embedding matrix, and model from Azure Blob Storage
#     download_blob_to_file(TOKENIZER_URL, "tokenizer.json")
#     download_blob_to_file(EMBEDDING_URL, "embedding_matrix.npy")
#     download_blob_to_file(MODEL_URL, "best_sentiment_analysis_model.pth")

#     # Load the tokenizer dictionary
#     with open("tokenizer.json", 'r') as f:
#         tokenizer = json.load(f)

#     # Load the saved embedding matrix
#     embedding_matrix = np.load("embedding_matrix.npy")
#     embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

#     # Initialize the model with the embedding matrix
#     vocab_size = len(tokenizer) + 1
#     model = SentimentLSTM(vocab_size, embedding_dim=embedding_matrix.shape[1], embedding_matrix=embedding_matrix)
    
#     # Load the saved model weights
#     model.load_state_dict(torch.load("best_sentiment_analysis_model.pth", map_location=torch.device('cpu')))
#     device = torch.device("cpu")
#     model.to(device)
#     model.eval()
    
#     return model, device

# def predict_sentiment(model, device, input_tensor):
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         probability = output.item()  # Get the probability score from the model
        
#         # Determine the sentiment label and confidence score
#         if probability >= 0.5:
#             sentiment_label = "Negative"
#             confidence_score = probability  # Confidence in it being Positive
#         else:
#             sentiment_label = "Positive"
#             confidence_score = 1 - probability  # Confidence in it being Negative

#     return sentiment_label, confidence_score


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import json
# from azure.storage.blob import BlobServiceClient
# from dotenv import load_dotenv
# import os
# import io

# load_dotenv()

# # Azure Blob Storage connection
# AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING").strip()
# MODEL_URL = os.getenv("MODEL_URL").strip()
# TOKENIZER_URL = os.getenv("TOKENIZER_URL").strip()
# EMBEDDING_URL = os.getenv("EMBEDDING_URL").strip()

# blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# def read_blob_to_memory(blob_url):
#     blob_client = blob_service_client.get_blob_client(container="project2", blob=blob_url.split('/')[-1])
#     return blob_client.download_blob().readall()

# # Define your model architecture (ensure it matches the training architecture)
# class SentimentLSTM(nn.Module):
#     def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, n_layers=2, embedding_matrix=None):
#         super(SentimentLSTM, self).__init__()

#         # Load pre-trained embeddings
#         if embedding_matrix is not None:
#             self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
#         else:
#             self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.3, bidirectional=True)
#         self.fc1 = nn.Linear(hidden_dim * 2, 128)  # Adding an extra dense layer
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.embedding(x)
#         lstm_out, _ = self.lstm(x)
#         lstm_out = lstm_out[:, -1, :]  # Take the output of the last timestep
#         x = F.dropout(F.relu(self.fc1(lstm_out)), p=0.3)  # Apply dropout with probability 0.3
#         x = self.fc2(x)
#         return self.sigmoid(x)

# def load_model():
#     # Read the tokenizer directly from Azure Blob Storage into memory
#     tokenizer_blob = read_blob_to_memory(TOKENIZER_URL)
#     tokenizer = json.loads(tokenizer_blob.decode("utf-8"))

#     # Read the embedding matrix directly from Azure Blob Storage into memory
#     embedding_blob = read_blob_to_memory(EMBEDDING_URL)
#     embedding_matrix = np.load(io.BytesIO(embedding_blob), allow_pickle=True)
#     embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

#     # Initialize the model with the embedding matrix
#     vocab_size = len(tokenizer) + 1
#     model = SentimentLSTM(vocab_size, embedding_dim=embedding_matrix.shape[1], embedding_matrix=embedding_matrix)
    
#     # Read the model weights directly from Azure Blob Storage into memory
#     model_blob = read_blob_to_memory(MODEL_URL)
#     model.load_state_dict(torch.load(io.BytesIO(model_blob), map_location=torch.device('cpu')))
    
#     device = torch.device("cpu")
#     model.to(device)
#     model.eval()
    
#     return model, device

# def predict_sentiment(model, device, input_tensor):
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         probability = output.item()  # Get the probability score from the model
        
#         # Determine the sentiment label and confidence score
#         if probability >= 0.5:
#             sentiment_label = "Negative"
#             confidence_score = probability  # Confidence in it being Positive
#         else:
#             sentiment_label = "Positive"
#             confidence_score = 1 - probability  # Confidence in it being Negative

#     return sentiment_label, confidence_score
