import sys
import json
from pathlib import Path

from PySide2.QtCore import Qt, Slot, Signal
from PySide2.QtGui import QIcon

from PySide2 import QtWidgets, QtCore

import numpy as np
import scipy.signal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt

import soundfile

_config = {
    "filename_A": None,
    "filename_B": None,
    "filename_C": None,
    "NFFT": 2048
}


def get_config():
    global _config
    return _config


def load_config() -> None:
    path = Path(__file__).parent / "config.json"
    if path.exists():
        with open(str(path), "r") as f:
            global _config
            _config = json.load(f)
    else:
        print("AlterFalter: use default settings")


def write_config() -> None:
    path = Path(__file__).parent / "config.json"
    with open(str(path), "w") as f:
        global _config
        json.dump(_config, f, indent=2)


def load_wave_file(path: str) -> np.ndarray:
    try:
        data, fs = soundfile.read(path)
        data = data.T
        if fs != 48000:
            # TODO 2021-01-07 funktion zum resamplen implementieren
            print(f"Cant read audio file at {path} with a samplerate of {fs}."
                  f"Expected samplerate: {48000}")
            return None
        return data
    except RuntimeError:
        print(f"file not found: {path}")
    return None


class WidgetSignal(QtWidgets.QWidget):

    calculationDesired = Signal()
    writeDone = Signal(str)

    def __init__(self, title: str, parent=None):
        super(WidgetSignal, self).__init__(parent)

        self.sig = np.zeros((2, 1024))
        self.fn_signal = ""

        _layout = QtWidgets.QVBoxLayout()

        # spectrogram view
        self.figure = Figure(figsize=(5, 5))
        self.figure.set_tight_layout(dict(pad=0.3))
        self.axes: plt.Axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.figure)
        _layout.addWidget(self.canvas)
        _layout.setStretchFactor(self.canvas, 1)
        _layout.addWidget(NavigationToolbar2QT(self.canvas, self))

        # file name
        self._group_filename = QtWidgets.QGroupBox(title)
        _layout.addWidget(self._group_filename)

        _layout_filename = QtWidgets.QVBoxLayout()
        _layout_filename_buttons = QtWidgets.QHBoxLayout()
        self._label_filename = QtWidgets.QLabel("Dateiname?")
        _layout_filename.addWidget(self._label_filename)

        _layout_filename_buttons = QtWidgets.QHBoxLayout()
        self._calc_button = QtWidgets.QPushButton("Calculate")
        self._calc_button.clicked.connect(self.calculationDesired.emit)
        _layout_filename_buttons.addWidget(self._calc_button)
        _layout_filename_buttons.addStretch()

        self._open_button = QtWidgets.QPushButton("Open")
        self._open_button.clicked.connect(self._on_filedialog_open_file)
        _layout_filename_buttons.addWidget(self._open_button)

        self._save_button = QtWidgets.QPushButton("Save")
        self._save_button.clicked.connect(self._on_filedialog_save_file)
        _layout_filename_buttons.addWidget(self._save_button)

        _layout_filename.addLayout(_layout_filename_buttons)
        self._group_filename.setLayout(_layout_filename)

        self.setLayout(_layout)

    def set_data(self, data):
        self.sig = data.copy()
        self.sig = np.atleast_2d(self.sig)
        self.update_plot()

    def update_plot(self):
        nfft = get_config()["NFFT"]
        f, t, STFT = scipy.signal.stft(self.sig, 48000, window='hann', nperseg=nfft)
        mag = np.abs(STFT)
        mag[mag < 1e-12] = 1e-12
        mag_log = 20*np.log10(mag)

        # self.figure.clear()

        self.axes.imshow(mag_log[0], aspect="auto", origin="lower", cmap="jet", interpolation="nearest")

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def open_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        self._label_filename.setText(fn_signal)
        data = load_wave_file(fn_signal)
        self.set_data(data)

    def _on_filedialog_open_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.wav", dir=str(pn_before))
        self._label_filename.setText(fn_signal)
        if fn_signal is not "":
            self.open_file(fn_signal)

    def save_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        soundfile.write(self.fn_signal, self.sig.T, 48000, subtype="PCM_32")
        self.writeDone.emit(self.fn_signal)

    def _on_filedialog_save_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getSaveFileName(filter="*.wav", dir=str(pn_before))
        self._label_filename.setText(fn_signal)
        if fn_signal is "":
            return
        if not fn_signal.endswith(".wav"):
            fn_signal = fn_signal + ".wav"
        if fn_signal is not "":
            self.save_file(fn_signal)


class MainAlterFalter(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alter Falter")
        app_icon = QIcon("icon.png")
        self.setWindowIcon(app_icon)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        layout = QtWidgets.QHBoxLayout()
        self.widgetSignalA = WidgetSignal("Signal A")
        self.widgetSignalA.calculationDesired.connect(self.calcA)
        self.widgetSignalA.writeDone.connect(self.writeA)
        layout.addWidget(self.widgetSignalA)

        _label_convolution = QtWidgets.QLabel("<h1>*</h1>")
        layout.addWidget(_label_convolution)

        self.widgetSignalB = WidgetSignal("Signal B")
        self.widgetSignalB.calculationDesired.connect(self.calcB)
        self.widgetSignalB.writeDone.connect(self.writeB)
        layout.addWidget(self.widgetSignalB)

        _label_assignment = QtWidgets.QLabel("<h1>=</h1>")
        layout.addWidget(_label_assignment)

        self.widgetSignalC = WidgetSignal("Signal C")
        self.widgetSignalC.calculationDesired.connect(self.calcC)
        self.widgetSignalC.writeDone.connect(self.writeC)
        layout.addWidget(self.widgetSignalC)

        self._main.setLayout(layout)

        # initial IR file loading
        fn_wave = get_config()["filename_A"]
        if fn_wave == None:
            fn_wave = str(Path(__file__).parent / "example/A.wav")
        self.widgetSignalA.open_file(fn_wave)

        fn_wave = get_config()["filename_B"]
        if fn_wave == None:
            fn_wave = str(Path(__file__).parent / "example/B.wav")
        self.widgetSignalB.open_file(fn_wave)

        fn_wave = get_config()["filename_C"]
        if fn_wave == None:
            fn_wave = str(Path(__file__).parent / "example/C.wav")
        self.widgetSignalC.open_file(fn_wave)

    def calcA(self):
        print("Calc A")

    def calcB(self):
        print("Calc B")

    def calcC(self):
        print("Calc C")
        self.widgetSignalA.sig

        len_c = self.widgetSignalA.sig.shape[-1] + self.widgetSignalB.sig.shape[-1]
        pad_a = len_c - self.widgetSignalA.sig.shape[-1]
        pad_b = len_c - self.widgetSignalB.sig.shape[-1]
        sig_a = np.pad(self.widgetSignalA.sig, ((0, 0), (0, pad_a)))
        sig_b = np.pad(self.widgetSignalB.sig, ((0, 0), (0, pad_b)))

        ffta = np.fft.rfft(sig_a)
        fftb = np.fft.rfft(sig_b)
        ffth = fftb * ffta
        h = np.fft.irfft(ffth)
        self.widgetSignalC.set_data(h)

    def writeA(self, fn_out: str):
        get_config()["filename_A"] = fn_out
        write_config()

    def writeB(self, fn_out: str):
        get_config()["filename_B"] = fn_out
        write_config()

    def writeC(self, fn_out: str):
        get_config()["filename_C"] = fn_out
        write_config()


if __name__ == "__main__":

    load_config()

    qapp = QtWidgets.QApplication(sys.argv)

    main = MainAlterFalter()
    main.show()

    sys.exit(qapp.exec_())
