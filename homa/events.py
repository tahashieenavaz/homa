import cv2
from .main import win
from inspect import signature

event_map = {
    "click":     cv2.EVENT_LBUTTONDOWN,
    "rclick":    cv2.EVENT_RBUTTONDOWN,
    "mclick":    cv2.EVENT_MBUTTONDOWN,
    "mouseup":   cv2.EVENT_LBUTTONUP,
    "rmouseup":  cv2.EVENT_RBUTTONUP,
    "dblclick":  cv2.EVENT_LBUTTONDBLCLK,
    "rdblclick": cv2.EVENT_RBUTTONDBLCLK,
    "move":      cv2.EVENT_MOUSEMOVE
}


def create_wrapper_function(event_name: str, handler: callable):
    def wrapper_function(event, x, y, flags, param):
        if event != event_map[event_name]:
            return

        argument_count = len(signature(handler).parameters)
        if argument_count == 2:
            args = (x, y)
        elif argument_count == 3:
            args = (x, y, flags)
        elif argument_count == 4:
            args = (x, y, flags, param)

        handler(*args)

    return wrapper_function


def onClick(key: str, handler: callable):
    win(key)
    cv2.setMouseCallback(key, create_wrapper_function("click", handler))


def onRightClick(key: str, handler: callable):
    win(key)
    cv2.setMouseCallback(key, create_wrapper_function("rclick", handler))


def onMouseUp(key: str, handler: callable):
    win(key)
    cv2.setMouseCallback(key, create_wrapper_function("mouseup", handler))


def onMove(key: str, handler: callable):
    win(key)
    cv2.setMouseCallback(key, create_wrapper_function("mousemove", handler))


def onDoubleClick(key: str, handler: callable):
    win(key)
    cv2.setMouseCallback(key, create_wrapper_function("dblclick", handler))
