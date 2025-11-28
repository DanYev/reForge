# rcmath Module: 32-bit and 64-bit Precision Support

## Overview

The `rcmath` module now supports both 32-bit (float) and 64-bit (double) precision computations. The implementation uses automatic dispatch based on input array dtype, providing both memory efficiency and numerical precision when needed.

## Architecture

```
reforge/rfgmath/
├── rcmath.py          # Python wrapper with automatic dispatch
├── rcymath32.pyx      # 32-bit (float) Cython implementation
└── rcymath64.pyx      # 64-bit (double) Cython implementation
```

## Key Changes

1. **rcymath64.pyx** (formerly rcmath.pyx)
   - Original 64-bit double precision implementation
   - Uses `double` (C) / `np.float64` (NumPy)

2. **rcymath32.pyx** (new)
   - 32-bit float precision implementation
   - Uses `float` (C) / `np.float32` (NumPy)
   - ~50% memory savings compared to 64-bit

3. **rcmath.py** (new)
   - Pure Python wrapper
   - Automatically dispatches to rcymath32 or rcymath64 based on input dtype
   - Provides unified API with optional dtype override

## Usage

### Automatic dispatch based on input dtype

```python
import numpy as np
from reforge.rfgmath import rcmath

# Use 64-bit precision (default)
vecs64 = np.random.rand(10, 3).astype(np.float64)
hessian64 = rcmath.hessian(vecs64, cutoff=1.2)
# -> uses rcymath64, returns float64

# Use 32-bit precision  
vecs32 = np.random.rand(10, 3).astype(np.float32)
hessian32 = rcmath.hessian(vecs32, cutoff=1.2)
# -> uses rcymath32, returns float32
```

### Force specific precision

```python
# Force 32-bit even with 64-bit input
vecs = np.random.rand(10, 3).astype(np.float64)
hessian32 = rcmath.hessian(vecs, cutoff=1.2, dtype=np.float32)
# -> converts input to float32, uses rcymath32
```

## Available Functions

All functions support both 32-bit and 64-bit precision:

- `calculate_hessian(resnum, x, y, z, cutoff, spring_constant, dd, dtype=None)`
- `hessian(vec, cutoff, spring_constant, dd, dtype=None)`
- `perturbation_matrix_old(covariance_matrix, resnum, dtype=None)`
- `perturbation_matrix(covariance_matrix, normalize, dtype=None)`
- `perturbation_matrix_par(covariance_matrix, normalize, dtype=None)`
- `perturbation_matrix_iso(ccf, normalize, dtype=None)`
- `perturbation_matrix_iso_par(ccf, normalize, dtype=None)`
- `td_perturbation_matrix(ccf, normalize, dtype=None)`
- `td_perturbation_matrix_par(ccf, normalize, dtype=None)`
- `covmat(X, dtype=None)`
- `pcovmat(X, dtype=None)`

## Compilation

Both Cython modules must be compiled:

```bash
python setup.py build_ext --inplace
```

This will compile:
- `reforge.rfgmath.rcymath32` from `rcymath32.pyx`
- `reforge.rfgmath.rcymath64` from `rcymath64.pyx`

## Performance Considerations

### Memory Usage
- **float32**: ~50% memory savings
- **float64**: Higher precision, more memory

### Numerical Precision
- **float32**: ~7 decimal digits of precision
- **float64**: ~15 decimal digits of precision

### Speed
- **float32**: Often faster due to:
  - Better cache utilization
  - Potential SIMD vectorization benefits
  - Less memory bandwidth required
  
- **float64**: May be faster on some operations where CPU has optimized 64-bit paths

### Recommendations

Use **float32** when:
- Working with large systems (memory constrained)
- Moderate precision is acceptable
- Speed is critical
- Doing exploratory analysis

Use **float64** when:
- Maximum numerical precision is required
- Working with ill-conditioned matrices
- Final production calculations
- Small systems where memory isn't a concern

## Example

See `examples/rcmath_dtype_example.py` for comprehensive usage examples.

## Migration Guide

### Old code (before changes)
```python
from reforge.rfgmath import rcmath  # This was the compiled .so

hessian = rcmath.hessian(vecs, cutoff=1.2)
```

### New code (after changes)
```python
from reforge.rfgmath import rcmath  # This is now the Python wrapper

# Same API - automatically uses float64 if vecs is float64
hessian = rcmath.hessian(vecs, cutoff=1.2)

# Or explicitly choose precision
hessian32 = rcmath.hessian(vecs, cutoff=1.2, dtype=np.float32)
hessian64 = rcmath.hessian(vecs, cutoff=1.2, dtype=np.float64)
```

The API is fully backward compatible!

## Error Handling

If a required precision module isn't compiled, you'll get a clear error:

```python
ImportError: rcymath32 module is not compiled. Cannot use float32 precision.
```

## Testing

After compilation, test both precisions:

```bash
python examples/rcmath_dtype_example.py
```

## Notes

- The wrapper automatically converts input arrays to the target dtype
- Output dtype matches the implementation used (float32 or float64)
- All decorators (@timeit, @memprofit) work with both versions
- OpenMP parallelization works with both versions
