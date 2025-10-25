import json


def settings(key: str, _cache: dict = {}):
    if not _cache:
        with open("settings.json", "r") as f:
            _cache.update(json.load(f))
    return _cache.get(key)


def get_settings(*args, **kwargs):
    return settings(*args, **kwargs)
