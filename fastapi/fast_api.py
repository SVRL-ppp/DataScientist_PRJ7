# Importing Necessary modules
from fastapi import FastAPI 
from components.model import model_prediction
from pydantic import BaseModel

# Declaring our FastAPI instance
app = FastAPI()

class ID(BaseModel):
    SKID: str

@app.get("/")
def home():
    return {"health_check":"OK"}

# Defining path operation for /name endpoint
#@app.post('/predict')
#async def predict(id: ID):
#    proba = model_prediction(id.SKID)
#    return {'proba':float(proba[0][0])}