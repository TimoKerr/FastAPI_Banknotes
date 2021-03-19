# FastAPI
This project is a template to use FastAPI for deployment. The data comes from a Kaggle competition where a classification engine needs to classify banknotes on whether they are genuine or fake, based on four input characteristics.

# venv
First create a virtual environment: ```python3 -m venv ~/.FastAPI_banknote```

# Install
```make install``` to install latest pip and packages according to requirements.txt

# Model training
The model used to make inference is trainer in the Jupyer Notebook. At this point it is a simply RandomForest classifier, but will be extended in the future. The model is saved as ```.pkl``` file to be called in the app.

# App 
Important to note that in the input type is controlled for by making a BankNote class in ```BankNotes.py```. This ensures informative error callbacks together with the Pydantic package. A Web GUI still needs to be implemented but is trivial.

# Run
The app can be run by ```uvicorn app:app --reload```.


