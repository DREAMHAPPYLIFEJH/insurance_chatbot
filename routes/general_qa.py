from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents import general_qa

router = APIRouter(prefix="/api/general_qa", tags=["general_qa"])


class GeneralQARequest(BaseModel):
    user_input: str


@router.post("")
def ask(req: GeneralQARequest):
    try:
        result = general_qa.run(req.user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
