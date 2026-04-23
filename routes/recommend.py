from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents import recommendation

router = APIRouter(prefix="/api/recommend", tags=["recommendation"])


class RecommendRequest(BaseModel):
    user_input: str


@router.post("")
def recommend(req: RecommendRequest):
    try:
        result = recommendation.run(req.user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
