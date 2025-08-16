# GPU Optimization

This repository contains a collection of CUDA kernels designed to execute a parallelized aeronautical Multidisciplinary Design Optimization (MDO) pipeline. The performance-focused implementation resulted in a system that can analyze 1 billion airplane variations to find an optimal solution within minutes.

---

## Technical Architecture

### Kernel Indexing and Execution Model
The kernels utilize a 3D grid layout to maximize occupancy across the GPU’s streaming multiprocessors. The global thread coordinates ($X, Y, Z$) are flattened into a unique 1D index ($idx$) to map threads to the parameter space:

$$\text{idx} = x + y \cdot \text{grid\\_width} + z \cdot \text{grid\\_width} \cdot \text{grid\\_height}$$

This $idx$ is then decomposed back into individual parameter coordinates using a backward modulo and division loop. This allows the system to sweep across $N$ dimensions of variations (e.g., Aspect Ratio, Wing Span, etc.) where each thread evaluates a single point in the $N$-dimensional grid.

### Numerical Methods and Memory
Numerical integration is implemented via a **Runge-Kutta 4th Order (RK4)** template function. The implementation (defined in the CUDA source) accepts a device-side **lambda expression** for the state derivative function ($dfunc$). This allows the solver to remain decoupled from the specific domain physics while benefiting from `__forceinline__` optimization.

Other specific optimizations include:
* **Operator Overloading**: Custom overloads for `float3` and `float4` types enable vector-like arithmetic operations within device code.
* **Stack Allocation**: Parameters are stored in stack-allocated arrays (`params[N_PARAMS]`) within the kernel to minimize global memory access latency.
* **Loop Unrolling**: Uses `#pragma unroll` to optimize the parameter generation and iteration loops, reducing instruction overhead.

---

## Implementation Details

### CUDA Kernels
* **single_kernel.cu**: Implements the `stage_single_kernel`. It performs the primary MDO calculation, including stability and performance analysis, in a single pass for simpler testing. It uses constant memory for system parameters (e.g., $rho$, $g\_acc$) and structured buffers for output.
* **visualization_kernel.cu**: A mapping kernel that tiles results into a 2D grayscale image buffer. It normalizes result values based on a global min/max and converts them to `uint8` for visual inspection of the parameter space topology.
* **stage1/2_kernel.cu**: Supports multi-stage execution where intermediate results are stored in GPU-resident buffers to avoid unnecessary host-to-device synchronization.

### Python Integration
The integration logic (demonstrated in `core-*.ipynb`) manages the following pipeline:
1. **Compilation**: `pycuda.compiler.SourceModule` compiles the `.cu` source files into GPU binaries at runtime.
2. **Memory Management**: Uses `drv.mem_alloc` for pre-allocating result buffers and `drv.memcpy_dtoh` (Device to Host) for retrieving final arrays after kernel execution.
3. **Dispatch**: Kernels are launched with calculated `grid` and `block` dimensions to ensure the number of threads matches the total number of parameter variations.

---

## Performance Considerations

The system is optimized for **Compute Unified Device Architecture (CUDA)** capable hardware (tested on T4/L4 architectures). By offloading the MDO evaluation to the GPU, the system achieves massive parallelism over the parameter grid, where each thread independently computes structural, aerodynamic, and performance criteria for a specific aircraft configuration. Data remains in GPU global memory between stages of the pipeline to minimize bandwidth bottlenecks.