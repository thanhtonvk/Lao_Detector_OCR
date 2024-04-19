from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from typing import Union

from entities.card_type import CardType
from entities.dimension import BoundingBox, Corners


class CardCropping(BaseModel):
    label: CardType = None
    score: float = None
    bbox: BoundingBox = None
    corners: Corners = None
    image: np.ndarray = None
    # points_name: list = None

    class Config:
        arbitrary_types_allowed = True


class IdCardCropping(CardCropping):
    pass


class PassportCardCropping(CardCropping):
    pass

class HouseholdCardCropping(CardCropping):
    pass

class HouseholdNextPrintCardCropping(CardCropping):
    pass

class HouseholdNextHandwritingCardCropping(CardCropping):
    pass