"""
FastAPI app that deploys model without web GUI. 
"""
import uvicorn
from fastapi import FastAPI
from mylib import BankNotes
import pickle

app = FastAPI()
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

# Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {"message": "This is a banknote classifier"}


# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post("/predict")
def predict_banknote(data: BankNotes.BankNote):
    data = data.dict()
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]
    entropy = data["entropy"]

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        output = f"Pred is {prediction[0]}, thus a fake note"
    else:
        output = f"Pred is {prediction[0]}, thus a genuine note"
    return {"prediction": output}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
