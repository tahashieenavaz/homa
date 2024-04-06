import cv2
from .classes.Logger import Logger
from .classes.Repository import Repository
import numpy
from .helpers.alias import repo


def destroy(key: str | None = None) -> None:
    if key is not None:
        cv2.destroyWindow(key)
        return

    cv2.destroyAllWindows()


def win(key: str):
    cv2.namedWindow(key)


def path(directory: str) -> None:
    Repository.directory = directory


def write(key: str, filename: str) -> None:
    cv2.imwrite(
        filename=filename,
        img=Repository.images[key]
    )


def save(*args, **kwargs) -> None:
    write(args, kwargs)


def image(filename: str, key: str | None = None, color: bool = True) -> None:
    # TODO: add no extension in the file
    if key is None:
        key = filename.split(".")[0]

    Repository.images[key] = cv2.imread(filename, int(color))
    return Repository.images[key]


def wait(delay=0):
    cv2.waitKey(delay)


def showWait(*args, **kwargs):
    kwargs["wait"] = True
    show(*args, **kwargs)


def show(*windows, **settings):
    for window in windows:
        window.show()

    if settings["wait"] == True:
        cv2.waitKey()


def black(key: str, width: int, height: int, channels=3):
    repo(
        key,
        numpy.zeros([height, width, channels], dtype="uint8")
    )


def white(key: str, width: int, height: int, channels=3):
    repo(
        key,
        numpy.ones([height, width, channels], dtype="uint8") * (2 ** 8 - 1)
    )


def setting(key: str, value: any = None) -> any:
    if value is not None:
        Repository.settings[key] = value
        return True

    setting_value = Repository.settings[key]
    return setting_value if setting_value else None


def refresh(key: str) -> None:
    cv2.imshow(Repository.windows[key], Repository.images[key])


def equipWithRefresh(function: callable) -> callable:
    def inner(*args, **kwargs):
        function(*args, **kwargs)
        refresh(args[0])

    return inner
