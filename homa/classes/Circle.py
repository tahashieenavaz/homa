from typing import Tuple
from ..helpers.alias import setting


class Circle(object):
    def __init__(self, x: int, y: int, radius: int, color: Tuple[int, int, int] | None = None, stroke: int | None = None) -> None:
        if stroke is None:
            stroke = setting("stroke")

        if color is None:
            color = setting("color")

        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.stroke = stroke
