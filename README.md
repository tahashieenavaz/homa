<p align="center">
	<img
		src="https://raw.githubusercontent.com/tahashieenavaz/homa/main/art/homa.svg"
		width=500
	/>
</p>

<hr />

Homa is an easy way to start learning Computer Vision with OpenCV.

## Loading Images

Images could be loaded with the `Image` class, that accepts the file name.

```python
from homa import *

horse = Image("horse.jpg"
show(horse, wait=True)
# or alternatively
showWait(horse)
```

## Smoothing

### Blur

```python
from homa import *

horse = Image("horse.jpg")
horse.blur(7) .   # using (7, 7) kernel
horse.blur(7, 19) # using (7, 19) kernel
showWait(horse)
```

### Gaussian Blur

```python
from homa import *

horse = Image("horse.jpg")
horse.gaussian(7) .   # using (7, 7) kernel
horse.gaussian(7, 19) # using (7, 19) kernel
showWait(horse)
```

### Median Blur

```python
from homa import *

horse = Image("horse.jpg")
horse.median(7)
```

## Stacking

```python
from homa import *

horse = Image("horse.jpg")
horse.blur(9, 7)

show(
	vstack(horse),
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
