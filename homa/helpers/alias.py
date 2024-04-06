from ..classes.Repository import Repository
from ..classes.Collection import Collection
from typing import List


def repo(key: str | None = None, value: any = None) -> any:
    if key is None and value is None:
        return Repository.images

    if key is not None and value is None:
        return Repository.addImage(key, value)

    Repository.addImage(key, value)
    return True


def collection(items: List[any]):
    return Collection(items)
