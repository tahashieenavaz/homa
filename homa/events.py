import cv2
from .main import win
from inspect import signature
from .classes.Repository import Repository

event_map = {
    "click":     cv2.EVENT_LBUTTONDOWN,
    "rclick":    cv2.EVENT_RBUTTONDOWN,
    "mclick":    cv2.EVENT_MBUTTONDOWN,
    "mouseup":   cv2.EVENT_LBUTTONUP,
    "rmouseup":  cv2.EVENT_RBUTTONUP,
    "dblclick":  cv2.EVENT_LBUTTONDBLCLK,
    "rdblclick": cv2.EVENT_RBUTTONDBLCLK,
    "mousemove": cv2.EVENT_MOUSEMOVE
}


def createMouseCallback(events: dict):
    def innerMouseCallback(event, x, y, flags, param):
        for e, handler in events.items():
            if event != e:
                continue

            argumentCount = len(signature(handler).parameters)
            if argumentCount == 2:
                args = (x, y)
            elif argumentCount == 3:
                args = (x, y, flags)
            elif argumentCount == 4:
                args = (x, y, flags, param)
            elif argumentCount == 1:
                args = (x,)
            elif args == 0:
                args = tuple()

            handler(*args)

    return innerMouseCallback
