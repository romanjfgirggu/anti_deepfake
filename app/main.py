import sys
import json
import mimetypes
from pathlib import Path
from typing import Any

import httpx
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QFormLayout,
    QGroupBox,
)


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg", ".mpg"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma"
}


def detect_media_type(file_path: str) -> str:
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(path.name)

    if mime_type:
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("audio/"):
            return "audio"

    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in AUDIO_EXTENSIONS:
        return "audio"

    return "unknown"


def normalize_response(data: dict[str, Any]) -> dict[str, Any]:

    label = (
        data.get("Predicted")
        or data.get("label")
        or data.get("prediction")
        or data.get("result")
    )

    score = (
        data.get("Confidence")
        or data.get("score")
        or data.get("confidence")
        or data.get("probability")
    )

    is_fake = None
    if isinstance(label, str):
        lowered = label.lower()
        if lowered in {"fake", "deepfake"}:
            is_fake = True
        elif lowered in {"real", "genuine"}:
            is_fake = False

    return {
        "label": label,
        "score": score,
        "is_fake": is_fake,
        "raw": data,
    }


class ApiClient:
    def __init__(
        self,
        base_url: str,
        video_endpoint: str,
        audio_endpoint: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.video_endpoint = self._normalize_endpoint(video_endpoint)
        self.audio_endpoint = self._normalize_endpoint(audio_endpoint) if audio_endpoint else None

    @staticmethod
    def _normalize_endpoint(endpoint: str | None) -> str | None:
        if not endpoint:
            return None
        endpoint = endpoint.strip()
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        return endpoint

    def _resolve_endpoint(self, media_type: str) -> str:
        if media_type == "video":
            if not self.video_endpoint:
                raise ValueError("Не задан video endpoint")
            return self.video_endpoint

        if media_type == "audio":
            if not self.audio_endpoint:
                raise ValueError(
                    "Audio endpoint пока не задан. "
                    "Когда добавишь backend для аудио, просто укажешь путь."
                )
            return self.audio_endpoint

        raise ValueError("Поддерживаются только video/audio файлы")

    def predict(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        media_type = detect_media_type(file_path)
        if media_type == "unknown":
            raise ValueError("Неподдерживаемый тип файла")

        endpoint = self._resolve_endpoint(media_type)

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "application/octet-stream"

        timeout = httpx.Timeout(10000.0, connect=200.0)

        with httpx.Client(base_url=self.base_url, timeout=timeout) as client:
            with open(path, "rb") as f:
                files = {
                    "file": (path.name, f, mime_type)
                }
                r = client.post("/v1/health")
                response = client.post(endpoint, files=files)

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"HTTP {e.response.status_code}\n{e.response.text}"
                ) from e

            try:
                payload = response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Backend вернул не JSON.\nТело ответа:\n{response.text}"
                ) from e

        return normalize_response(payload)


class PredictionWorker(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        base_url: str,
        video_endpoint: str,
        audio_endpoint: str,
        file_path: str,
    ):
        super().__init__()
        self.base_url = base_url
        self.video_endpoint = video_endpoint
        self.audio_endpoint = audio_endpoint
        self.file_path = file_path

    def run(self):
        try:
            client = ApiClient(
                base_url=self.base_url,
                video_endpoint=self.video_endpoint,
                audio_endpoint=self.audio_endpoint or None,
            )
            result = client.predict(self.file_path)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Anti-Deepfake Desktop Client")
        self.setMinimumWidth(760)

        self.thread: QThread | None = None
        self.worker: PredictionWorker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout()
        self.setLayout(root)

        title = QLabel("Anti-Deepfake Desktop Client")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(title)

        api_group = QGroupBox("Настройки API")
        api_form = QFormLayout()

        self.base_url_input = QLineEdit("http://127.0.0.1:8000")
        self.video_endpoint_input = QLineEdit("/v1/detect/video")
        self.audio_endpoint_input = QLineEdit("/v1/detect/audio")  # future-ready

        api_form.addRow("Base URL:", self.base_url_input)
        api_form.addRow("Video endpoint:", self.video_endpoint_input)
        api_form.addRow("Audio endpoint:", self.audio_endpoint_input)
        api_group.setLayout(api_form)
        root.addWidget(api_group)

        file_group = QGroupBox("Файл")
        file_layout = QHBoxLayout()

        self.file_input = QLineEdit()
        self.file_input.setReadOnly(True)
        self.file_input.setPlaceholderText("Выбери video/audio файл")

        self.browse_button = QPushButton("Выбрать файл")
        self.browse_button.clicked.connect(self.select_file)

        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.browse_button)
        file_group.setLayout(file_layout)
        root.addWidget(file_group)

        self.predict_button = QPushButton("Получить предикт")
        self.predict_button.clicked.connect(self.start_prediction)
        self.predict_button.setStyleSheet(
            "padding: 10px; font-size: 15px; font-weight: 600;"
        )
        root.addWidget(self.predict_button)

        self.status_label = QLabel("Статус: ожидание")
        self.status_label.setStyleSheet("font-size: 14px;")
        root.addWidget(self.status_label)

        result_group = QGroupBox("Результат")
        result_layout = QVBoxLayout()

        self.summary_label = QLabel("Пока нет результата")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-size: 14px;")

        self.raw_output = QTextEdit()
        self.raw_output.setReadOnly(True)
        self.raw_output.setPlaceholderText("Сюда будет выведен raw JSON от backend")

        result_layout.addWidget(self.summary_label)
        result_layout.addWidget(self.raw_output)
        result_group.setLayout(result_layout)
        root.addWidget(result_group)

    def select_file(self):
        file_filter = (
            "Media files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.mpeg *.mpg "
            "*.mp3 *.wav *.ogg *.flac *.aac *.m4a *.wma);;"
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm *.flv *.mpeg *.mpg);;"
            "Audio files (*.mp3 *.wav *.ogg *.flac *.aac *.m4a *.wma);;"
            "All files (*)"
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выбор media файла",
            "",
            file_filter,
        )

        if file_path:
            self.file_input.setText(file_path)
            media_type = detect_media_type(file_path)
            self.status_label.setText(f"Статус: выбран {media_type} файл")

    def start_prediction(self):
        file_path = self.file_input.text().strip()
        base_url = self.base_url_input.text().strip()
        video_endpoint = self.video_endpoint_input.text().strip()
        audio_endpoint = self.audio_endpoint_input.text().strip()

        if not file_path:
            QMessageBox.warning(self, "Ошибка", "Сначала выбери файл")
            return

        if not base_url:
            QMessageBox.warning(self, "Ошибка", "Укажи Base URL")
            return

        media_type = detect_media_type(file_path)
        if media_type == "unknown":
            QMessageBox.warning(self, "Ошибка", "Поддерживаются только audio/video файлы")
            return

        if media_type == "audio" and not audio_endpoint:
            QMessageBox.warning(self, "Ошибка", "Для audio пока не задан endpoint")
            return

        self.predict_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.status_label.setText("Статус: выполняется запрос...")
        self.summary_label.setText("Ожидаем ответ модели...")
        self.raw_output.clear()

        self.thread = QThread(self)
        self.worker = PredictionWorker(
            base_url=base_url,
            video_endpoint=video_endpoint,
            audio_endpoint=audio_endpoint,
            file_path=file_path,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        self.worker.finished.connect(self.on_prediction_success)
        self.worker.error.connect(self.on_prediction_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.finished.connect(self._cleanup_after_thread)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()


    def _cleanup_after_thread(self):
        self.predict_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.worker = None
        self.thread = None


    def on_prediction_success(self, result: dict[str, Any]):
        self.predict_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.status_label.setText("Статус: готово")

        label = result.get("label")
        score = result.get("score")
        is_fake = result.get("is_fake")

        parts = []

        if label is not None:
            parts.append(f"Класс: {label}")

        if score is not None:
            try:
                percent = float(score) * 100
                parts.append(f"Confidence: {percent:.2f}%")
            except (TypeError, ValueError):
                parts.append(f"Confidence: {score}")

        if is_fake is not None:
            parts.append("Вердикт: FAKE" if is_fake else "Вердикт: REAL")

        if not parts:
            parts.append("Ответ получен, но стандартные поля не распознаны")

        self.summary_label.setText("\n".join(parts))
        self.raw_output.setPlainText(
            json.dumps(result.get("raw", {}), indent=2, ensure_ascii=False)
        )

        self.worker = None
        self.thread = None

    def on_prediction_error(self, error_text: str):
        self.predict_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.status_label.setText("Статус: ошибка")
        self.summary_label.setText("Не удалось получить предикт")
        self.raw_output.setPlainText(error_text)

        QMessageBox.critical(self, "Ошибка запроса", error_text)

        self.worker = None
        self.thread = None


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()