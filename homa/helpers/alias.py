from ..classes.Repository import Repository
from ..classes.Collection import Collection
from typing import List
from typing import Tuple


def repo(key: str | None = None, value: any = None) -> any:
    if key is None and value is None:
        return Repository.images

    if key is not None and value is None:
        return Repository.addImage(key, value)

    Repository.addImage(key, value)
    return True


def collection(items: List[any] | Tuple[any]):
    return Collection(items)

def setting(key: str, value: any = None) -> any:
    if value is not None:
        Repository.settings[key] = value
        return True

    setting_value = Repository.settings[key]
    return setting_value if setting_value else None