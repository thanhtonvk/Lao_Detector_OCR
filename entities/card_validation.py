from pydantic import BaseModel

from entities.card_type import CardType


class CardValidation(BaseModel):
    label: CardType
    score: float
    is_valid: bool
