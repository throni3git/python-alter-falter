import sys
import json
from pathlib import Path

from PySide2.QtCore import Qt, Slot, Signal
from PySide2.QtGui import QIcon, QFontMetrics

from PySide2 import QtWidgets, QtCore

import numpy as np
import scipy.signal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt

import soundfile
import sounddevice as sd

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
        if fs != 48000:
            # TODO 2021-01-07 funktion zum resamplen implementieren
            print(f"Cant read audio file at {path} with a samplerate of {fs}."
                  f"Expected samplerate: {48000}")
            return None
        return data
    except RuntimeError:
        print(f"file not found: {path}")
    return None


def filter50_20k(x, sr):  # filters everything outside out 20 - 20000 Hz
    nyq = 0.5 * sr
    sos = scipy.signal.butter(2, [50.0 / nyq, 20000.0 / nyq], btype='band', output='sos')
    return scipy.signal.sosfiltfilt(sos, x)


class WidgetSignal(QtWidgets.QWidget):

    calculationDesired = Signal()
    signalFilenameChanged = Signal(str)

    def __init__(self, title: str, parent=None):
        super(WidgetSignal, self).__init__(parent)

        self.sig = np.zeros((2, 1024))
        self.fn_signal = ""
        self.fs = 48000
        self.which_chan = 0

        _layout = QtWidgets.QVBoxLayout(self)

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

        _layout_filename = QtWidgets.QVBoxLayout(self._group_filename)
        _layout_filename_buttons = QtWidgets.QHBoxLayout()
        self._lineedit_filename = QtWidgets.QTextEdit("Dateiname?")
        f = self._lineedit_filename.font()
        f.setPointSize(f.pointSize() * 0.9)
        self._lineedit_filename.setFont(f)
        self._set_filename_text("Dateiname?")
        self._lineedit_filename.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._lineedit_filename.setReadOnly(True)
        _layout_filename.addWidget(self._lineedit_filename)

        _layout_filename_buttons = QtWidgets.QHBoxLayout()
        self._calc_button = QtWidgets.QPushButton("Calculate")
        self._calc_button.clicked.connect(self.calculationDesired.emit)
        _layout_filename_buttons.addWidget(self._calc_button)
        _layout_filename_buttons.addStretch()

        self._play_button = QtWidgets.QPushButton("Play")
        self._play_button.clicked.connect(self._toggle_play_stop)
        _layout_filename_buttons.addWidget(self._play_button)

        self._open_button = QtWidgets.QPushButton("Open")
        self._open_button.clicked.connect(self._on_filedialog_open_file)
        _layout_filename_buttons.addWidget(self._open_button)

        self._save_button = QtWidgets.QPushButton("Save")
        self._save_button.clicked.connect(self._on_filedialog_save_file)
        _layout_filename_buttons.addWidget(self._save_button)

        _layout_filename.addLayout(_layout_filename_buttons)

    def set_data(self, data):
        self.sig = data.copy()
        self.set_channel(self.which_chan)
        self.update_plot()

    def set_channel(self, which_chan):
        if which_chan > 2 or which_chan < 0:
            raise NotImplementedError("only left or right channels supported")
        which_chan = which_chan if self.num_channels == 2 else 0
        self.which_chan = which_chan

    def update_plot(self):
        nfft = get_config()["NFFT"]
        f, t, STFT = scipy.signal.stft(self.sig[self.which_chan], self.fs, window='hann', nperseg=nfft)
        mag = np.abs(STFT)
        floor = 1e-12
        mag[mag < floor] = floor
        mag_log = 20*np.log10(mag)
        vmin = 20*np.log10(floor)
        vmax = 0

        self.axes.imshow(mag_log,
                         aspect="auto",
                         origin="lower",
                         cmap="jet",
                         interpolation="nearest",
                         vmin=vmin,
                         vmax=vmax)
        self.axes.grid(True, alpha=0.3)

        self.axes.set_xlabel("time in s")
        if t[-1] < 10:
            step = 0.5
            step_type = np.float32
        else:
            step = int(np.ceil(t[-1] / 20))
            step_type = np.int32
        self.axes.set_xticks(np.arange(0, t[-1], step) * self.fs * 2 / nfft)
        self.axes.set_xticklabels(np.arange(0, t[-1], step, dtype=step_type))

        self.axes.set_ylabel("freq in kHz")
        self.axes.set_yticks(np.arange(0, f[-1], 1000) * nfft / self.fs)
        self.axes.set_yticklabels(np.arange(0, f[-1], 1000, dtype=np.int32)//1000)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def _set_filename_text(self, text: str):
        self._lineedit_filename.setText(text)

        font = self._lineedit_filename.font()
        fontMetrics = QFontMetrics(font)
        textSize = fontMetrics.size(0, text)
        textHeight = textSize.height() + 26  # Need to tweak
        self._lineedit_filename.setMaximumHeight(textHeight)

    def open_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        self._set_filename_text(fn_signal)

        data = load_wave_file(fn_signal)
        if data is None:
            return
        self.num_channels = data.shape[1] if data.ndim == 2 else 1
        if self.num_channels > 2:
            raise NotImplementedError("only mono or stereo")
        data = np.atleast_2d(data)
        # transpose, so the shape is (channels, samples)
        if self.num_channels == 2:
            data = data.T
        self.set_data(data)
        self.signalFilenameChanged.emit(fn_signal)

    def _on_filedialog_open_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.wav", dir=str(pn_before))
        self._set_filename_text(fn_signal)
        if fn_signal is not "":
            self.open_file(fn_signal)

    def _toggle_play_stop(self):
        sd.play(self.sig.T, self.fs)

    def save_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        soundfile.write(self.fn_signal, self.sig.T, 48000, subtype="PCM_32")
        self.signalFilenameChanged.emit(self.fn_signal)

    def _on_filedialog_save_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getSaveFileName(filter="*.wav", dir=str(pn_before))
        self._set_filename_text(fn_signal)
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
        container_layout = QtWidgets.QVBoxLayout(self._main)

        layout = QtWidgets.QHBoxLayout()
        container_layout.addLayout(layout)
        self.widgetSignalA = WidgetSignal("Signal A")
        self.widgetSignalA.calculationDesired.connect(self.calcA)
        self.widgetSignalA.signalFilenameChanged.connect(self.signalFilenameChangedA)
        layout.addWidget(self.widgetSignalA)

        _label_convolution = QtWidgets.QLabel("<h1>*</h1>")
        layout.addWidget(_label_convolution)

        self.widgetSignalB = WidgetSignal("Signal B")
        self.widgetSignalB.calculationDesired.connect(self.calcB)
        self.widgetSignalB.signalFilenameChanged.connect(self.signalFilenameChangedB)
        layout.addWidget(self.widgetSignalB)

        _label_assignment = QtWidgets.QLabel("<h1>=</h1>")
        layout.addWidget(_label_assignment)

        self.widgetSignalC = WidgetSignal("Signal C")
        self.widgetSignalC.calculationDesired.connect(self.calcC)
        self.widgetSignalC.signalFilenameChanged.connect(self.signalFilenameChangedC)
        layout.addWidget(self.widgetSignalC)

        layout_options = QtWidgets.QHBoxLayout()
        container_layout.addLayout(layout_options)
        self.radio_which_channel_left = QtWidgets.QRadioButton("left")
        self.radio_which_channel_right = QtWidgets.QRadioButton("right")

        self.radio_which_channel_group = QtWidgets.QButtonGroup()
        self.radio_which_channel_group.buttonClicked.connect(self.check_which_channel)
        self.radio_which_channel_group.addButton(self.radio_which_channel_left)
        self.radio_which_channel_group.addButton(self.radio_which_channel_right)
        layout_options.addWidget(self.radio_which_channel_left)
        layout_options.addWidget(self.radio_which_channel_right)
        self.radio_which_channel_left.setChecked(True)
        layout_options.addStretch()

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

    def check_which_channel(self, radio_button):
        if radio_button.text() == "left":
            self.widgetSignalA.set_channel(0)
            self.widgetSignalB.set_channel(0)
            self.widgetSignalC.set_channel(0)
        else:
            self.widgetSignalA.set_channel(1)
            self.widgetSignalB.set_channel(1)
            self.widgetSignalC.set_channel(1)

        self.widgetSignalA.update_plot()
        self.widgetSignalB.update_plot()
        self.widgetSignalC.update_plot()

    def calcA(self):
        print("Calc A")

    def calcB(self):
        print("Calc B")

        sig_a = self.widgetSignalA.sig
        sig_c = self.widgetSignalC.sig

        max_channels = max(self.widgetSignalA.num_channels, self.widgetSignalC.num_channels)
        if self.widgetSignalA.num_channels != max_channels:
            sig_a = np.vstack((sig_a, sig_a))
        if self.widgetSignalC.num_channels != max_channels:
            sig_c = np.vstack((sig_c, sig_c))

        len_A = self.widgetSignalA.sig.shape[-1]
        len_C = sig_c.shape[-1]

        if len_C < len_A:
            sig_c = np.pad(sig_c, ((0, 0), (0, len_A - len_C)))
            len_C = len_A

        len_diff = len_C - len_A
        sig_A = np.pad(sig_a, ((0, 0), (0, len_diff)))
        full_A = np.pad(sig_A, ((0, 0), (0, len_C)))
        full_C = np.pad(sig_c, ((0, 0), (0, len_C)))

        # sweep_padded = padarray(sweep_raw, sweep_duration_samples*2, before=sr*0)
        # rec_padded = padarray(rec_raw, sweep_duration_samples*2, before=sr*0)

        full_C = filter50_20k(full_C, 48000)

        ffta = np.fft.rfft(full_A)
        fftc = np.fft.rfft(full_C)

        # bin_lower = int(100/48000*2 * fftc.shape[1])
        # ramp_lower = 0.5-0.5*np.cos(np.arange(bin_lower)/bin_lower*np.pi)
        # ramp_lower = np.atleast_2d(ramp_lower)**4
        # ramp_lower = np.tile(ramp_lower, (fftc.shape[0] // ramp_lower.shape[0], 1))
        # fftc[:, :bin_lower] *= ramp_lower

        # bin_upper = int(22000/48000*2 * fftc.shape[1])
        # ramp_upper = 0.5+0.5*np.cos(np.arange(fftc.shape[1]-bin_upper)/(fftc.shape[1]-bin_upper)*np.pi)
        # ramp_upper = np.atleast_2d(ramp_upper)**4
        # ramp_upper = np.tile(ramp_upper, (fftc.shape[0] // ramp_upper.shape[0], 1))
        # fftc[:, bin_upper:] *= ramp_upper

        # fftc[:, :bin_lower] = 0
        # fftc[:, bin_upper:] = 0

        ffta[np.abs(ffta) == 0] = 1e-12
        ffth = fftc / ffta
        h1 = np.fft.irfft(ffth)
        # h1 = filter50_20k(h1, sr)
        h1 = h1[..., :h1.shape[-1]//2]

        self.widgetSignalB.num_channels = max_channels
        self.widgetSignalB.set_data(h1)

    def calcC(self):
        print("Calc C")

        len_c = self.widgetSignalA.sig.shape[-1] + self.widgetSignalB.sig.shape[-1]
        pad_a = len_c - self.widgetSignalA.sig.shape[-1]
        pad_b = len_c - self.widgetSignalB.sig.shape[-1]
        sig_a = np.pad(self.widgetSignalA.sig, ((0, 0), (0, pad_a)))
        sig_b = np.pad(self.widgetSignalB.sig, ((0, 0), (0, pad_b)))

        max_channels = max(self.widgetSignalA.num_channels, self.widgetSignalB.num_channels)
        if self.widgetSignalA.num_channels != max_channels:
            sig_a = np.vstack((sig_a, sig_a))
        if self.widgetSignalB.num_channels != max_channels:
            sig_b = np.vstack((sig_b, sig_b))

        ffta = np.fft.rfft(sig_a)
        fftb = np.fft.rfft(sig_b)
        ffth = fftb * ffta
        h = np.fft.irfft(ffth)
        self.widgetSignalC.num_channels = max_channels
        self.widgetSignalC.set_data(h)

    def signalFilenameChangedA(self, fn_out: str):
        get_config()["filename_A"] = fn_out
        write_config()

    def signalFilenameChangedB(self, fn_out: str):
        get_config()["filename_B"] = fn_out
        write_config()

    def signalFilenameChangedC(self, fn_out: str):
        get_config()["filename_C"] = fn_out
        write_config()


if __name__ == "__main__":

    load_config()

    qapp = QtWidgets.QApplication(sys.argv)

    main = MainAlterFalter()
    main.show()

    sys.exit(qapp.exec_())
