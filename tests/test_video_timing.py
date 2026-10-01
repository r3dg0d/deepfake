import json
import shutil
import subprocess

import numpy as np
import pytest

from deepfake.outputs.sinks import FFmpegVideoSink


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg required")
def test_short_render_retains_final_frame_and_bounds_audio(tmp_path):
    audio = tmp_path / "audio.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=3", str(audio)], check=True
    )
    target = tmp_path / "output.mkv"
    sink = FFmpegVideoSink(target, 60, (64, 64), audio_from=audio, encoder="libx264")
    for index in range(120):
        sink.write(np.full((64, 64, 3), index, np.uint8))
    sink.close()
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_entries",
                "stream=codec_type,nb_read_frames:format=duration",
                "-of",
                "json",
                str(target),
            ]
        )
    )
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_read_frames"]) == 120
    assert float(info["format"]["duration"]) <= 2.06
