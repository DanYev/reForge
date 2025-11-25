# AsyncHeavyReporter - Comprehensive Testing & Benchmarking

A complete testing and benchmarking suite for the `AsyncHeavyReporter` - an OpenMM reporter that performs heavy calculations in parallel with the MD engine.

## 📁 Files Overview

### Core Implementation
- **`reforge/mdsystem/async_reporter.py`** - Main async reporter implementation

### Testing Suite
- **`tests/test_async_reporter.py`** - Comprehensive unit tests
- **`tests/benchmark_async_reporter.py`** - Performance benchmarking suite
- **`tests/run_reporter_tests.py`** - Unified test runner script

### Examples
- **`examples/async_reporter_examples.py`** - Usage examples and patterns

## 🚀 Quick Start

### Run Tests
```bash
# Run all unit tests
cd tests
python run_reporter_tests.py --tests

# Or use pytest directly
pytest test_async_reporter.py -v
```

### Run Benchmarks
```bash
# Quick benchmark (fast, ~30 seconds)
python run_reporter_tests.py --bench

# Full benchmark suite (comprehensive, ~5-10 minutes)
python run_reporter_tests.py --bench-full

# Direct execution
python benchmark_async_reporter.py          # Quick
python benchmark_async_reporter.py --full   # Full suite
```

### Run Everything
```bash
python run_reporter_tests.py --all
```

## 📊 Test Coverage

### Unit Tests (`test_async_reporter.py`)

#### TestAsyncReporterBasics
- ✅ Module import
- ✅ Reporter initialization
- ✅ `describeNextReport()` interface
- ✅ Thread startup

#### TestAsyncReporterFunctionality
- ✅ Single report processing
- ✅ Multiple report processing
- ✅ Heavy calculation handling
- ✅ Data copying and queuing

#### TestAsyncReporterThreadSafety
- ✅ Concurrent report submission
- ✅ Queue overflow handling
- ✅ Thread-safe result storage
- ✅ Race condition prevention

#### TestAsyncReporterErrorHandling
- ✅ Calculation function errors
- ✅ Empty finalization
- ✅ Double finalization
- ✅ Graceful degradation

#### TestAsyncReporterOutputs
- ✅ File creation
- ✅ NPZ format (dict results)
- ✅ NPY format (array results)
- ✅ Data integrity

#### TestAsyncReporterIntegration
- ✅ Realistic use cases
- ✅ Distance calculations
- ✅ Center of mass tracking
- ✅ End-to-end workflows

## ⚡ Benchmark Suite

### Benchmark Configurations

The full benchmark suite tests:

| Config | Atoms | Frames | Calc Time | Description |
|--------|-------|--------|-----------|-------------|
| 1 | 100 | 20 | 0.01s | Small system, fast calc |
| 2 | 100 | 20 | 0.1s | Small system, slow calc |
| 3 | 1000 | 50 | 0.05s | Medium system, medium calc |
| 4 | 1000 | 100 | 0.1s | Medium system, slow calc |
| 5 | 5000 | 20 | 0.2s | Large system, very slow calc |

### Metrics Measured

- **Total Report Time** - Total time for all report() calls
- **Average Report Time** - Time per individual report
- **Throughput** - Frames processed per second
- **Speedup** - Async vs Sync performance ratio
- **Overhead** - Additional time compared to theoretical minimum

### Expected Results

Typical speedup factors:
- **Fast calculations (0.01s)**: ~1.5-2x speedup
- **Medium calculations (0.05-0.1s)**: ~3-5x speedup  
- **Slow calculations (0.2s+)**: ~5-10x speedup

**Key insight**: The slower the calculation, the bigger the benefit of async processing!

## 📈 Benchmark Output

The benchmark suite generates:

1. **Console Output** - Real-time progress and results
2. **Summary Table** - Comparison of all configurations
3. **Plots** (`benchmark_results.png`):
   - Total report time comparison
   - Throughput comparison
   - Speedup factors
   - Average time per report
4. **JSON Results** (`benchmark_results.json`) - Raw data for analysis

### Example Output

```
==================================================================
BENCHMARK SUMMARY
==================================================================
Reporter         Atoms    Frames   Calc(s)    Total(s)   Avg(s)     FPS        Speedup   
------------------------------------------------------------------
Sync             100      20       0.010      0.245      0.0123     81.63      1.00x     
Async            100      20       0.010      0.152      0.0076     131.58     1.61x     
------------------------------------------------------------------
Sync             1000     100      0.100      10.234     0.1023     9.77       1.00x     
Async            1000     100      0.100      2.145      0.0214     46.62      4.77x     
------------------------------------------------------------------
```

## 🔬 Usage Examples

### Basic Usage
```python
from reforge.mdsystem.async_reporter import AsyncHeavyReporter

def my_calculation(data):
    """Heavy calculation function"""
    positions = data['positions']
    # Expensive computation here
    return {'step': data['step'], 'result': result}

# Create reporter
reporter = AsyncHeavyReporter(
    'output.npz',
    reportInterval=1000,
    calculation_func=my_calculation,
    queue_size=20
)

# Add to simulation
simulation.reporters.append(reporter)

# Run simulation (MD engine runs at full speed!)
simulation.step(100000)

# Finalize (wait for all calculations)
reporter.finalize()
```

### Advanced Examples

See `examples/async_reporter_examples.py` for:
- Distance matrix calculation
- Contact map tracking
- Radius of gyration monitoring
- Hydrogen bond analysis
- Multiple simultaneous reporters

## 🧪 Testing Guidelines

### For Developers

#### Adding New Tests
```python
# In test_async_reporter.py
class TestNewFeature:
    def test_my_feature(self, temp_output_dir, mock_simulation):
        reporter = AsyncHeavyReporter(...)
        # Test code
        assert ...
```

#### Adding New Benchmarks
```python
# In benchmark_async_reporter.py
def new_benchmark_scenario():
    benchmark = ReporterBenchmark()
    benchmark.benchmark_reporter(
        AsyncHeavyReporter, "MyTest",
        n_atoms=1000, n_frames=50, calculation_time=0.05
    )
```

### Running Specific Tests
```bash
# Single test class
pytest test_async_reporter.py::TestAsyncReporterBasics -v

# Single test method
pytest test_async_reporter.py::TestAsyncReporterBasics::test_initialization -v

# Tests matching pattern
pytest test_async_reporter.py -k "thread" -v
```

## 📋 Continuous Integration

### Recommended CI Configuration

```yaml
# .github/workflows/test_reporter.yml
name: Test AsyncReporter
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install pytest numpy matplotlib openmm
      - name: Run tests
        run: |
          cd tests
          pytest test_async_reporter.py -v
      - name: Run quick benchmark
        run: |
          cd tests
          python benchmark_async_reporter.py
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Tests fail with "Queue full" warnings
- **Solution**: Increase `queue_size` in test configurations

**Issue**: Benchmark shows no speedup
- **Solution**: Calculation might be too fast. Increase `calculation_time`

**Issue**: Tests timeout
- **Solution**: Reduce `n_frames` or calculation complexity

**Issue**: Thread doesn't stop
- **Solution**: Always call `reporter.finalize()` in finally block

## 📊 Performance Tips

1. **Queue Size**: Set to ~2x the number of frames between slow calculations
2. **Report Interval**: Balance between data density and overhead
3. **Calculation Time**: If calc takes < 0.001s, async overhead may exceed benefit
4. **Multiple Reporters**: Each runs independently, perfect for different analyses

## 🔍 Understanding Results

### When to Use Async Reporter

✅ **Use when:**
- Calculations take > 0.01s per frame
- You want continuous analysis during simulation
- MD trajectory generation is important
- Running multiple independent analyses

❌ **Don't use when:**
- Calculations are trivial (< 0.001s)
- Can do post-processing after simulation
- Memory constrained (queue buffering)
- Need guaranteed real-time results

## 📚 References

- **OpenMM Reporters**: http://docs.openmm.org/latest/api-python/library.html#reporters
- **Threading in Python**: https://docs.python.org/3/library/threading.html
- **Queue Module**: https://docs.python.org/3/library/queue.html

## 🤝 Contributing

When contributing reporter improvements:
1. Add unit tests in `test_async_reporter.py`
2. Update benchmarks if performance characteristics change
3. Document new features in examples
4. Run full test suite before submitting

## 📄 License

Same as parent reForge project.

---

**Questions?** Check the examples or run the benchmarks to see the reporter in action!
