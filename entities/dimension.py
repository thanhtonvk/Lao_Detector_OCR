from typing import Union, Sequence
from pydantic import BaseModel


IntOrFloat = Union[int, float]


class Point(BaseModel):
    x: float = None
    y: float = None


class BoundingBox(BaseModel):
    top_left: Point = None
    bot_right: Point = None

    def as_list(self):
        if self.top_left and self.bot_right:
            return [self.top_left.x, self.top_left.y, self.bot_right.x, self.bot_right.y]
        return []


class Corners(BaseModel):
    top_left: Point = None
    top_right: Point = None
    bot_right: Point = None
    bot_left: Point = None
    center: Point = None

    def as_list(self):
        _list = []
        if self.top_left and self.top_right and self.bot_right and self.bot_left:
            _list = [
                self.top_left.dict().values(),
                self.top_right.dict().values(),
                self.bot_right.dict().values(),
                self.bot_left.dict().values(),
            ]
            if self.center:
                _list.append(self.center.dict().values())
        return _list


def point_from_list(arr: Sequence[IntOrFloat]) -> Point:
    if len(arr) < 2:
        raise AttributeError("Invalid length of list")
    return Point(x=arr[0], y=arr[1])


def bbox_from_list(arr: Sequence[IntOrFloat]) -> BoundingBox:
    if len(arr) < 4:
        raise AttributeError("Invalid length of list")
    return BoundingBox(
        top_left=Point(x=arr[0], y=arr[1]),
        bot_right=Point(x=arr[2], y=arr[3])
    )
