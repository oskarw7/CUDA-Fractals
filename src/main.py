from cuda.mandelbrot_julia import generate_fractal
from app import run_app

# Used for testing
def explicit_generation():
    generate_fractal(-2.0, -1.0, 1.0, 1.0, 7680, 5120, 1000)  # 3:2 scale for the best fit
    generate_fractal(-2.25, -1.5, 2.25, 1.5, 7680, 5120, 1000, 'julia', 0.285, 0.01)

if __name__ == '__main__':
    run_app()