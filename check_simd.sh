#!/bin/bash
# Check if SIMD instructions are present in compiled Cython modules

echo "Checking for SIMD instructions in compiled modules..."
echo ""

for module in reforge/rfgmath/rcymath64*.so reforge/rfgmath/rcymath32*.so; do
    if [ -f "$module" ]; then
        echo "==================================="
        echo "Module: $module"
        echo "==================================="
        
        echo ""
        echo "AVX instructions found:"
        objdump -d "$module" | grep -E "vaddpd|vmulpd|vsqrtpd|vaddps|vmulps|vsqrtps" | head -20
        
        echo ""
        echo "SSE instructions found:"
        objdump -d "$module" | grep -E "addpd|mulpd|sqrtpd|addps|mulps|sqrtps" | head -20
        
        echo ""
        echo "Summary:"
        avx_count=$(objdump -d "$module" | grep -cE "v[a-z]+pd|v[a-z]+ps" || echo "0")
        sse_count=$(objdump -d "$module" | grep -cE "[a-z]+pd|[a-z]+ps" || echo "0")
        echo "  AVX/AVX2 instructions: $avx_count"
        echo "  SSE instructions: $sse_count"
        echo ""
    fi
done

echo ""
echo "To get detailed vectorization report, recompile with:"
echo "  python setup.py build_ext --inplace"
echo ""
echo "Look for messages like 'loop vectorized' in the output"
