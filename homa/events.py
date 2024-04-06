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
    "mousemove":      cv2.EVENT_MOUSEMOVE
}


def fire(key: str):
    events = Repository.events[key]
    finalHandler = createFinalHandler(events)

    win(key)
    cv2.setMouseCallback(key, finalHandler)


def createFinalHandler(events):
    def finalHandlerFunction(event, x, y, flags, param):
        for e in events:
            if event == event_map[e["event"]]:
                argument_count = len(signature(e["handler"]).parameters)
                if argument_count == 0:
                    args = tuple()
                elif argument_count == 2:
                    args = (x, y)
                elif argument_count == 3:
                    args = (x, y, flags)
                elif argument_count == 4:
                    args = (x, y, flags, param)

                e["handler"](*args)
    return finalHandlerFunction


def createWrapperFunction(event_name: str, handler: callable):
    def wrapperFunction(event, x, y, flags, param):
        if event != event_map[event_name]:
            return

        argument_count = len(signature(handler).parameters)
        if argument_count == 0:
            args = tuple()
        elif argument_count == 2:
            args = (x, y)
        elif argument_count == 3:
            args = (x, y, flags)
        elif argument_count == 4:
            args = (x, y, flags, param)

        handler(*args)

    return wrapperFunction


def createEvent(name: str, handler):
    return {
        "event": name,
        "handler": handler
    }
