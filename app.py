"""
FastAPI app that deploys model without web GUI. 
"""
import uvicorn
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from mylib import BankNotes
import pickle

app = FastAPI()

pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)


app.mount(
    "/static", StaticFiles(directory="static"), name="static"
)  # Mount static files, specify directory


templates = Jinja2Templates(directory="templates")  # Template Directory


@app.get("/data/{data}")
async def read_data(request: Request, data: str):
    return templates.TemplateResponse("index.html", {"request": request, "data": data})


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
