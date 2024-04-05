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


def showMany(*keys):
    for key in keys[0:-1]:
        show(key)

    showWait(keys[-1])


def show(key: any = None, wait: bool = False, window: str = "Homa Window") -> None:
    # TODO: add functionality to distinguish between camera and images

    if key is not None and not isinstance(key, str):
        Repository.imshow(window, key)

    elif key is None:
        for key, image in Repository.images.items():
            Repository.imshow(key, image)
            Repository.windows[key] = key

    elif key is not None:
        if key in Repository.images:
            Repository.imshow(key, Repository.images[key])
            Repository.windows[key] = key
        else:
            Logger.danger(f"No image found with key {key}")

    if wait:
        cv2.waitKey(0)


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
