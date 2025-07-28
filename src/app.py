from PySide6.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from src.cuda.mandelbrot import generate_fractal
import sys #temp



class FractalsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractals")



app = QApplication(sys.argv)
window = FractalsApp()
sys.exit(app.exec())