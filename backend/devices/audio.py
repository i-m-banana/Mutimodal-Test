from __future__ import annotations

from .base import DeviceBase


class AudioDevice(DeviceBase):
    def __init__(self, *, format=None, channels=1, rate=44100, chunk=1024) -> None:
        import pyaudio

        super().__init__()
        self.format = format or pyaudio.paInt16
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.audio = pyaudio.PyAudio()
        self.audio_stream: pyaudio.PyAudio.Stream | None = None

    def read(self):
        assert self.audio_stream is not None, "on_start not called"
        data = self.audio_stream.read(self.chunk)
        return True, data

    def on_start(self):
        import pyaudio

        self.audio_stream: pyaudio.PyAudio.Stream | None = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def on_done(self):
        assert self.audio_stream is not None, "on_start not called"
        self.audio_stream.stop_stream()
        self.audio_stream.close()
        self.audio.terminate()
