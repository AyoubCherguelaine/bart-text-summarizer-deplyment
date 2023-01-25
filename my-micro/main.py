from fastapi import FastAPI
from .model import model
from pydantic import BaseModel

app = FastAPI()

class text(BaseModel):
    text:str

@app.get("/")
async def read_root():
    return model.info.state


@app.get("/summarize")
async def read_item(text:text):
    
    return {"text":model.summarizer(text.text)}
