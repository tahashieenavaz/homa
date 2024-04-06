import cv2
from .classes.Repository import Repository


def destroy(key: str | None = None) -> None:
    if key is not None:
        cv2.destroyWindow(key)
        return

    cv2.destroyAllWindows()


def show(*windows, **settings):
    for window in windows:
        window.show()

    if settings["wait"] == True:
        cv2.waitKey()


def showWait(*args, **kwargs):
    kwargs["wait"] = True
    show(*args, **kwargs)


def setting(key: str, value: any = None) -> any:
    if value is not None:
        Repository.settings[key] = value
        return True

    setting_value = Repository.settings[key]
    return setting_value if setting_value else None
