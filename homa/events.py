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
