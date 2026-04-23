from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents import product_info

router = APIRouter(prefix="/api/product_info", tags=["product_info"])


class ProductInfoRequest(BaseModel):
    user_input: str


@router.post("")
def get_product_info(req: ProductInfoRequest):
    try:
        result = product_info.run(req.user_input)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
