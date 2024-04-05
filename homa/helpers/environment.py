import sys
from ..classes.Repository import Repository


def is_colab() -> bool:
    return 'google.colab' in sys.modules


def setting(key: str, value: any = None) -> any:
    if value is not None:
        Repository[key] = value
        return True

    setting_value = Repository.settings[key]
    return setting_value if setting_value else None
