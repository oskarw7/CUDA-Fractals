import matplotlib.pyplot as plt
import numpy as np
from numba import cuda


# Helper function executed on GPU
@cuda.jit(device=True)
def mandelbrot_pixel(x, y, iters_threshold):
    const = complex(x, y)
    z = 0.0j
    for i in range(iters_threshold):
        z = z ** 2 + const
        if abs(z) >= 2:
            return i
    return iters_threshold


# NOTE: Consider adding argtypes argument to JIT
# Kernel function
@cuda.jit
def create_fractal(min_x, min_y, max_x, max_y, image, iters_threshold):
    width = image.shape[1]
    height = image.shape[0]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    start_x, start_y = cuda.grid(2)
    step_x = cuda.gridDim.x * cuda.blockDim.x
    step_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, step_x):
        re = min_x + x * pixel_size_x
        for y in range(start_y, height, step_y):
            im = min_y + y * pixel_size_y
            image[y, x] = mandelbrot_pixel(re, im, iters_threshold)


# Function called from outer scope
def generate_fractal(image_width, image_height, iters_threshold):
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    block_dim = (32, 16)  # NOTE: Try different values
    grid_dim = (
        (image_width + block_dim[0] - 1) // block_dim[0],  # ceiling
        (image_height + block_dim[1] - 1) // block_dim[1]
    )

    device_image = cuda.to_device(image)
    create_fractal[grid_dim, block_dim](-2.0, -1.0, 1.0, 1.0, device_image, iters_threshold)
    cuda.synchronize()
    device_image.copy_to_host(image)

    dpi = 100
    figsize = (
        image_width / dpi,
        image_height / dpi
    )
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image, cmap='viridis', extent=(-2.0, 1.0, -1.0, 1.0))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('assets/mandelbrot.png')