import sys
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QMainWindow, QApplication, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QMenuBar, QMenu, QMessageBox,
    QProgressDialog, QFileDialog
)
from PySide6.QtGui import QAction
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import imsave
from cuda.mandelbrot_julia import generate_fractal


# Custom progress dialog
# Needed to prevent user from closing dialog and generating again while thread is still working
class StoppableProgressDialog(QProgressDialog):
    closed = Signal()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


# Background worker responsible for image generation
class FractalWorker(QThread):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self, min_x, min_y, max_x, max_y, width, height, iterations, mode, re_c, im_c):
        super().__init__()
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.width = width
        self.height = height
        self.iterations = iterations
        self.mode = mode
        self.re_c = re_c
        self.im_c = im_c

    def run(self):
        try:
            result = generate_fractal(
                self.min_x, self.min_y, self.max_x, self.max_y,
                self.width, self.height, self.iterations,
                self.mode, self.re_c, self.im_c
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# Main app class, responsible for creating GUI and managing events
class FractalsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fractals')
        self.fractal_mode = 'mandelbrot'
        self.current_image = None
        self.worker = None

        menu_bar = QMenuBar()
        file_menu = QMenu('File', self)
        save_action = QAction('Save Image...', self)
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        menu_bar.addMenu(file_menu)

        fractal_menu = QMenu('Generate fractal', self)
        self.mandelbrot_action = QAction('Mandelbrot', self)
        self.julia_action = QAction('Julia', self)
        self.mandelbrot_action.triggered.connect(self.set_mandelbrot)
        self.julia_action.triggered.connect(self.set_julia)
        fractal_menu.addAction(self.mandelbrot_action)
        fractal_menu.addAction(self.julia_action)
        menu_bar.addMenu(fractal_menu)

        help_menu = QMenu('Help', self)
        help_action = QAction('About Fields', self)
        help_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(help_action)
        menu_bar.addMenu(help_menu)

        self.setMenuBar(menu_bar)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.fig = Figure(figsize=(6, 6))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        main_layout.addWidget(self.canvas, stretch=3)

        control_layout = QVBoxLayout()
        self.inputs = {}

        for label, default in [
            ('Min X', '-2.0'), ('Max X', '2.0'),
            ('Min Y', '-1.5'), ('Max Y', '1.5'),
            ('Width', '1536'), ('Height', '1024'),
            ('Iterations', '100'),
        ]:
            control_layout.addWidget(QLabel(label))
            line_edit = QLineEdit(default)
            line_edit.setMaximumWidth(100)
            self.inputs[label] = line_edit
            control_layout.addWidget(line_edit)

        self.julia_re_label = QLabel('Re(c)')
        self.julia_re_input = QLineEdit('-0.73')
        self.julia_im_label = QLabel('Im(c)')
        self.julia_im_input = QLineEdit('0.19')
        for widget in [self.julia_re_label, self.julia_re_input, self.julia_im_label, self.julia_im_input]:
            control_layout.addWidget(widget)
            widget.setMaximumWidth(100)

        self.gen_button = QPushButton('Generate')
        self.gen_button.clicked.connect(self.generate_and_draw)
        control_layout.addWidget(self.gen_button)
        control_layout.addStretch()

        main_layout.addLayout(control_layout, stretch=1)
        self.set_mandelbrot()

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def set_mandelbrot(self):
        self.fractal_mode = 'mandelbrot'
        self.julia_re_label.hide()
        self.julia_re_input.hide()
        self.julia_im_label.hide()
        self.julia_im_input.hide()

    def set_julia(self):
        self.fractal_mode = 'julia'
        self.julia_re_label.show()
        self.julia_re_input.show()
        self.julia_im_label.show()
        self.julia_im_input.show()

    def generate_and_draw(self):
        try:
            min_x = float(self.inputs['Min X'].text())
            max_x = float(self.inputs['Max X'].text())
            min_y = float(self.inputs['Min Y'].text())
            max_y = float(self.inputs['Max Y'].text())
            width = int(self.inputs['Width'].text())
            height = int(self.inputs['Height'].text())
            iterations = int(self.inputs['Iterations'].text())
            re_c = float(self.julia_re_input.text()) if self.fractal_mode == 'julia' else 0.0
            im_c = float(self.julia_im_input.text()) if self.fractal_mode == 'julia' else 0.0
        except ValueError:
            QMessageBox.critical(self, 'Input Error', 'Invalid input type')
            return

        if width <= 0 or width >= 8000 or height <= 0 or height >= 8000:
            QMessageBox.critical(self, 'Input Error', 'Invalid resolution')
            return
        if iterations <= 0:
            QMessageBox.critical(self, 'Input Error', 'Invalid iterations threshold')
            return

        self.loading_dialog = StoppableProgressDialog("Generating fractal...", None, 0, 0, self)
        self.loading_dialog.setWindowTitle("Please Wait")
        self.loading_dialog.setWindowModality(Qt.ApplicationModal)
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.closed.connect(self.on_loading_dialog_closed)
        self.loading_dialog.show()

        self.worker = FractalWorker(
            min_x, min_y, max_x, max_y,
            width, height, iterations,
            self.fractal_mode, re_c, im_c
        )
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_loading_dialog_closed(self):
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            self.worker = None

    def on_generation_finished(self, image):
        self.loading_dialog.close()
        self.current_image = image
        self.ax.clear()
        self.ax.imshow(image, cmap='viridis')
        self.ax.axis('off')
        self.ax.set_position((0, 0, 1, 1))
        self.canvas.draw()
        self.worker = None

    def on_generation_error(self, message):
        self.loading_dialog.close()
        QMessageBox.critical(self, "Generation Error", message)
        self.worker = None

    def save_image(self):
        if self.current_image is None:
            QMessageBox.critical(self, "Save Image", "No image generated yet to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;All Files (*)")
        if file_path:
            try:
                imsave(file_path, self.current_image, cmap='viridis')
                QMessageBox.information(self, "Save Image", f"Image saved to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Image", f"Failed to save image:\n{e}")

    def show_help_dialog(self):
        help_text = """
        <b>Input Fields Explanation:</b><br><br>
        <ul>
          <li><b>Min X / Max X:</b> The real axis range for fractal calculation.</li>
          <li><b>Min Y / Max Y:</b> The imaginary axis range for fractal calculation.</li>
          <li><b>Width / Height:</b> Size of the output image in pixels.</li>
          <li><b>Iterations:</b> Maximum iterations per point (higher = more detail, slower).</li>
          <li><b>Julia Re / Julia Im:</b> Real and imaginary parts of the constant used in Julia sets.</li>
          <li><b>Fractal Mode:</b> Choose between Mandelbrot or Julia fractals.</li>
        </ul>
        """
        QMessageBox.information(self, "Help - Input Fields", help_text)


def run_app():
    app = QApplication(sys.argv)
    window = FractalsApp()
    window.show()
    sys.exit(app.exec())
