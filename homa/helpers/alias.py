from ..classes.Repository import Repository
from ..classes import Collection
from typing import List


def repo(key: str|None = None):
    if key is None:
        return Repository.images

    return Repository.images[key]


def collection(items: List[any]):
    return Collection(items)
