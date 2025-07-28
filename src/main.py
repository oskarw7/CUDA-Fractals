from cuda.mandelbrot import generate_fractal


def main():
   generate_fractal(-2.0, -1.0, 1.0, 1.0, 7680, 5120, 1000) # 3:2 scale for the best fit

if __name__ == '__main__':
    main()