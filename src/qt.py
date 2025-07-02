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
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHBoxLayout,
    QMessageBox,
)

from stt import WhisperSTT
from main import main


class DMWorker(QThread):
    """Runs the blocking DM loop in its own thread.

    Receives player input via a queue and emits DM speech, the new scene-image
    path, and the raw player text back to the GUI using the *dm_ready* signal.
    """

    dm_ready = pyqtSignal(str, str)  # history, image_path
    input_requested = pyqtSignal()

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
            self.input_requested.emit()
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

        self.record_btn = QPushButton('Start Recording')
        self.record_btn.setFixedWidth(100)
        self.record_btn.setFixedHeight(50)
        self.record_btn.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.record_btn)
        btn_row.addStretch()

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
        self.worker.input_requested.connect(self.on_input_requested)
        self.worker.start()

        # ========================= State & Connections =====================
        self.recording: bool = False
        self.record_btn.clicked.connect(self.toggle_recording)

        # Provide a status bar for user feedback if one doesn’t already exist
        if self.statusBar() is None:
            self.setStatusBar(QStatusBar(self))

    # ------------------------------------------------------------------ UI
    def toggle_recording(self) -> None:
        """Toggle between start and stop recording states."""
        if not self.recording:
            # -------- Begin recording --------
            try:
                self.stt_model.start()
            except RuntimeError as exc:
                QMessageBox.critical(self, 'Recording Error', str(exc))
                return
            self.recording = True
            self.record_btn.setText('Stop Recording')
            self.statusBar().showMessage('Recording…')  # type: ignore[attr-defined]
        else:
            # -------- End recording ----------
            self.recording = False
            self.record_btn.setEnabled(False)  # Disable until DM is ready
            self.statusBar().clearMessage()  # type: ignore[attr-defined]

            try:
                player_text = self.stt_model.close()
            except RuntimeError as exc:
                QMessageBox.warning(self, 'Error', str(exc))
                return

            self.record_btn.setText('Start Recording')

            self.worker.push_player_text(player_text)
            self.history.append(f'<b>Player:</b> {player_text}')

    # ---------------------------------------------------------------- slots
    def on_dm_ready(self, history: str, image_path: str) -> None:  # type: ignore[override]
        """Update the UI with the DM's response and re‑enable recording."""
        history = history.replace('**DM:**', '<b>DM:</b>').replace('**Player:**', '<b>Player:</b>')
        self.history.setText(history)
        self.load_image(Path(image_path))

    def on_input_requested(self) -> None:
        """Request player input from the user."""
        self.record_btn.setEnabled(True)

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
