from pydantic import BaseModel
from typing import List
import numpy as np

from entities.card_type import CardType
from entities.dimension import BoundingBox


class CardClassification(BaseModel):
    label: CardType = None
    score: float = None
    bbox: BoundingBox = None
    bboxes: List[BoundingBox] = None
    crop_img: np.ndarray = None

    class Config:
        arbitrary_types_allowed = True
