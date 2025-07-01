"""
PyQt6 GUI front‑end for the voice‑driven D&D framework.

Key features
------------
* Shows campaign title at top.
* Displays the most recent scene image (and automatically updates when a new one is generated).
* Conversation history pane (read‑only) shows alternating Player / DM lines.
* Start Recording / Stop Recording buttons wrap the WhisperSTT recorder.
* Background worker thread runs dm_turn + image_generation so GUI never blocks.
* Signals / slots used to send player input to worker and push DM output back to UI.
"""

from __future__ import annotations

import sys
import queue
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMessageBox,
)

from stt import WhisperSTT
from wip2 import main


class DMWorker(QThread):
    """Runs the blocking DM loop in its own thread.

    Receives player input via a queue and emits DM speech, the new scene-image
    path, and the raw player text back to the GUI using the *dm_ready* signal.
    """

    dm_ready = pyqtSignal(str, str, str)  # history, image_path

    def __init__(self, parent: Optional[QObject] | None = None) -> None:
        super().__init__(parent)
        self.queue: queue.Queue[str] = queue.Queue()

    # -------------------------------------------------- public API for the GUI
    def push_player_text(self, txt: str) -> None:
        """Add freshly‑transcribed player input to the processing queue."""
        self.queue.put(txt)

    # -------------------------------------------------------------- QThread API
    def run(self) -> None:  # noqa: D401 – Qt naming convention
        def player_input() -> str:
            return self.queue.get()

        for update in main(player_input):
            self.dm_ready.emit(update.history, str(update.image))


class MainWindow(QMainWindow):
    """Main GUI window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Voice‑Driven D&D')
        self.resize(1000, 800)

        # ============================= Widgets ==============================
        self.image_label = QLabel('No image yet')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(QSize(640, 360))

        self.history = QTextEdit()
        self.history.setReadOnly(True)

        self.start_btn = QPushButton('Start Recording')
        self.stop_btn = QPushButton('Stop Recording')
        self.stop_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label, stretch=3)
        layout.addWidget(self.history, stretch=2)
        layout.addLayout(btn_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ======================== Audio / Workers ==========================
        self.stt_model = WhisperSTT(model_name='base')

        self.worker = DMWorker()
        self.worker.dm_ready.connect(self.on_dm_ready)
        self.worker.start()

        # ========================= Connections =============================
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)

    # ------------------------------------------------------------------ UI
    def start_recording(self) -> None:
        """Begin capturing player speech."""
        try:
            self.stt_model.start()
        except RuntimeError as exc:
            QMessageBox.critical(self, 'Recording Error', str(exc))
            return
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage('Recording…')  # type: ignore[attr-defined]

    def stop_recording(self) -> None:
        """Stop recording, transcribe, and hand text to the DM worker."""
        try:
            player_text = self.stt_model.close()
        except RuntimeError as exc:
            QMessageBox.warning(self, 'Error', str(exc))
            return

        self.worker.push_player_text(player_text)
        self.history.append(f'<b>Player:</b> {player_text}')

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.statusBar().clearMessage()  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- slots
    def on_dm_ready(self, history: str, image_path: str) -> None:  # type: ignore[override]
        """Update the UI with the DM's response and the new scene image."""
        self.history.setText(history)
        self.load_image(Path(image_path))
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().clearMessage()  # type: ignore[attr-defined]

    # ---------------------------------------------------------------- helpers
    def load_image(self, path: Path) -> None:
        if not path.exists():
            return
        pixmap = QPixmap(str(path)).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.image_label.setText('')

    # -------------------------------------------------------------- cleanup
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.worker.terminate()
        self.worker.wait()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry‑point helper
# ---------------------------------------------------------------------------


def gui_main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    gui_main()
