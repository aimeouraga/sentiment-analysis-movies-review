from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from model import load_model, predict_sentiment
from preprocessing import preprocess_text
from auth import Token, authenticate_user, User, get_current_active_user, create_access_token
from datetime import timedelta
from dotenv import load_dotenv
import os
load_dotenv()


# Initialize FastAPI
app = FastAPI()

# Load the model and tokenizer
model, device = load_model()


ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Define the input data structure using Pydantic
class ReviewInput(BaseModel):
    review: str

# input_tensor = preprocess_text("bAD MOVIE NO LIKE")
        
# # Make the prediction
# sentiment_label, probability = predict_sentiment(model, device, input_tensor)

# Define the prediction endpoint
@app.post("/predict")
def predict(data: ReviewInput, current_user: User = Depends(get_current_active_user)):
    try:
        # Preprocess the review text
        input_tensor = preprocess_text(data.review)
        
        # Make the prediction
        sentiment_label, probability = predict_sentiment(model, device, input_tensor)
        
        # Return the result
        return {"review": data.review, "sentiment": sentiment_label, "Model Confidence Score": probability}
    
    except Exception as e:
        return {"error": str(e), "message": "An error occurred during the prediction process."}
    

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=int(ACCESS_TOKEN_EXPIRE_MINUTES))
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


@app.get("/health")
def health_check():
    return {"status": "API is running", "model_loaded": model is not None}
