<p align="center">
	<img
		src="https://raw.githubusercontent.com/tahashieenavaz/homa/main/art/homa.svg"
		width=500
	/>
</p>

<hr />

Homa is an easy way to start learning Computer Vision with OpenCV.

## Loading Images

Images could be loaded with the `image` helper, that accepts the file name and a key for the repository.

```python
from homa import *

image("horse.jpg", "horse")
```

Alternatively, following code will load the file into the repository with a key of everything before the last in the filename.

```python
from homa import *

image("horse.jpg") # stored as "horse"
```

## Smoothing

### Blur

```python
from homa import *

image("horse.jpg")

blur("horse", 7)                    # rewrites "horse" key
blur("horse", (7, 19))              # rewrites "horse" key
blur("horse", 9, "blurred horse")   # as a new key in the repository

showWait("blurred horse")
```

### Gaussian Blur

```python
from homa import *

image("horse.jpg")

gaussian("horse", 7)                             # rewrites "horse" key
gaussian("horse", (7, 19))                       # rewrites "horse" key
gaussian("horse", 9, "gaussian blurred horse")   # as a new key in the repository

showWait("gaussian blurred horse")
```

## Stacking

```python
from homa import *

image("horse.jpg")
blur("horse", 9, "blurred horse")

show(
	vstack("horse", "blurred horse"),
	window="Vstacked"
)

showWait(
	hstack("horse", "blurred horse"),
	window="Hstacked"
)
```

## Camera

Camera frames could be access from the repository with a key of `camera`.

```python
from homa import *

for _ in camera():
	show("camera")
```

You can simply combine camera frames with the supported effects.

```python
from homa import *

for _ in camera():
	blur("camera", 13, "blurred camera")

	show(
		vstack("camera", "blurred camera"),
		window="Camera Effect"
	)
```
