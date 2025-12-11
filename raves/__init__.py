"""
This lets you import RAVES functions in your own Python scripts, using one or more of the following lines:
```
from raves import raves
from raves import compute_ART
from raves import compute_MoDART
```
Assuming that you cloned the repository somewhere that your Python script can see.

N.B.: Unless you manually set the argument `multiprocess_pool_size=1`,
 the functions `raves` and `compute_ART` make use of multiprocessing,
 so you need to call them inside a `if __name__ == '__main__'` scope:
```
if __name__ == '__main__':
    raves("path/to/your/environment/folder")
```
"""
from .api import raves
from .src.compute_ART import compute_ART
from .src.compute_MoDART import compute_MoDART

__all__ = ["raves", "compute_ART", "compute_MoDART"]
