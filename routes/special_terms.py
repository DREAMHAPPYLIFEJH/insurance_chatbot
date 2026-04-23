from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from agents import special_terms
from langchain_community.chat_message_histories import ChatMessageHistory

router = APIRouter(prefix="/api/special_terms", tags=["special_terms"])

# 세션별 히스토리 저장
_histories: dict[str, ChatMessageHistory] = {}


class SpecialTermsRequest(BaseModel):
    session_id:     str
    user_input:     str
    current_source: Optional[str] = None
    current_terms:  Optional[list] = Field(default_factory=list)


@router.post("")
def get_special_terms(req: SpecialTermsRequest):
    try:
        if req.session_id not in _histories:
            _histories[req.session_id] = ChatMessageHistory()

        result = special_terms.run(
            user_input     = req.user_input,
            chat_history   = _histories[req.session_id],
            current_source = req.current_source,
            current_terms  = req.current_terms,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{session_id}")
def clear_history(session_id: str):
    _histories.pop(session_id, None)
    return {"status": "cleared"}
