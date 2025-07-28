import matplotlib.pyplot as plt
import numpy as np
from numba import cuda
import time


# Helper function executed on GPU
@cuda.jit(device=True)
def mandelbrot_pixel(re, im, iters_threshold):
    const = complex(re, im)
    z = 0.0j
    for i in range(iters_threshold):
        z = z ** 2 + const
        if abs(z) >= 2:
            return i
    return iters_threshold

# Helper function executed on GPU
@cuda.jit(device=True)
def julia_pixel(re, im, iters_threshold, const_re, const_im):
    const = complex(const_re, const_im)
    z = complex(re, im)
    for i in range(iters_threshold):
        z = z ** 2 + const
        if abs(z) >= 2:
            return i
    return iters_threshold

# NOTE: Consider adding argtypes argument to JIT
# Kernel function
@cuda.jit
def create_fractal(min_x, min_y, max_x, max_y, image, iters_threshold, mode_flag, const_re, const_im):
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
            if mode_flag == 1:
                image[y, x] = mandelbrot_pixel(re, im, iters_threshold)
            elif mode_flag == 0:
                image[y, x] = julia_pixel(re, im, iters_threshold, const_re, const_im)

# Helper function for converting matrix into image and saving it
def save_fractal(min_x, min_y, max_x, max_y, image, mode):
    dpi = 100
    figsize = (
        image.shape[1] / dpi,
        image.shape[0] / dpi
    )
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(image, cmap='viridis', extent=(min_x, max_x, min_y, max_y))
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    timestamp = time.strftime("%H:%M:%S_%d.%m.%Y")
    plt.savefig(f'../assets/{mode}_{timestamp}.png')

# Function called from outer scope
def generate_fractal(min_x, min_y, max_x, max_y, image_width, image_height, iters_threshold, mode='mandelbrot', const_re=0, const_im=0):
    image = np.zeros((image_height, image_width), dtype=np.uint8)

    block_dim = (32, 16)  # NOTE: Try different values
    grid_dim = (
        (image_width + block_dim[0] - 1) // block_dim[0],  # ceiling
        (image_height + block_dim[1] - 1) // block_dim[1]
    )

    device_image = cuda.to_device(image)
    mode_flag = 1 if mode == 'mandelbrot' else 0
    create_fractal[grid_dim, block_dim](min_x, min_y, max_x, max_y, device_image, iters_threshold, mode_flag, const_re, const_im)
    cuda.synchronize()
    device_image.copy_to_host(image)

    #save_fractal(min_x, min_y, max_x, max_y, image, mode)
    return image
