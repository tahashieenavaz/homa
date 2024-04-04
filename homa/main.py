import cv2
from .classes.Logger import Logger
from .classes.Repository import Repository


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


def show(key: any = None, wait: bool = False, window: str = "Homa Window") -> None:
    # TODO: add functionality to distinguish between camera and images

    if key is not None and not isinstance(key, str):
        Repository.imshow(window, key)
        Repository.windows[key] = window

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


def refresh(key: str) -> None:
    cv2.imshow(Repository.windows[key], Repository.images[key])
