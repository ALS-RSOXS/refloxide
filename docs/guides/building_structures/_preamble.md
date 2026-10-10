```python
from refloxide.model import MaterialSLD, FreeUniTensor, Mix, ReflectModel
from refloxide.profiles import (
    Diffusion,
    SecondOrderTransition,
    DepthProfile,
    Spline,
    Polynomial,
    CallableField,
    CallableDepthProfile,
)

vac = MaterialSLD("", 0, name="vacuum")(0, 0)
si = MaterialSLD("Si", 2.33, name="si")(0, 3.0)
mat = FreeUniTensor(energies=[284.4], name="film")
mat_a = FreeUniTensor(energies=[284.4], name="a")
mat_b = FreeUniTensor(energies=[284.4], name="b")
mat_c = FreeUniTensor(energies=[284.4], name="c")
```
