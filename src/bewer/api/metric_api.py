from fastapi import FastAPI, Body
from pydantic import BaseModel
from bewer import Dataset

app = FastAPI()

class TranscriptData(BaseModel):
    reference: str
    hypothesis: str

@app.post("/compute")
def compute(data: TranscriptData = Body()):
    dataset = Dataset()
    dataset.add(ref=data.reference, hyp=data.hypothesis)
    result = dataset[0].metrics.levenshtein.ops
    return result.to_json()
