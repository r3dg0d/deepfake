from deepfake.devices import VideoDevice, format_devices, list_v4l2_devices, resolve_cuda


def test_format_empty():
    text = format_devices([])
    assert "no /dev/video" in text.lower() or "not found" in text.lower()


def test_format_rows():
    rows = [
        VideoDevice(0, "/dev/video0", "USB Camera", "capture"),
        VideoDevice(10, "/dev/video10", "deepfake Loopback", "output"),
    ]
    text = format_devices(rows)
    assert "/dev/video0" in text
    assert "Loopback" in text


def test_list_v4l2_does_not_crash():
    # May be empty on this box; must not raise
    devices = list_v4l2_devices()
    assert isinstance(devices, list)


def test_resolve_cuda_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    # force path without assuming torch
    assert resolve_cuda("cpu") == "cpu"


def test_find_loopback_prefers_deepfake_label(monkeypatch):
    from deepfake import devices

    monkeypatch.setattr(
        devices,
        "list_v4l2_devices",
        lambda: [
            devices.VideoDevice(0, "/dev/video0", "USB Camera", "capture"),
            devices.VideoDevice(9, "/dev/video9", "Dummy video device (0x0000)", "output"),
            devices.VideoDevice(10, "/dev/video10", "deepfake", "output"),
        ],
    )
    monkeypatch.setattr(devices, "is_loopback", lambda i: i in (9, 10))
    assert devices.find_loopback_device() == "/dev/video10"
    monkeypatch.setattr(devices, "is_loopback", lambda i: False)
    monkeypatch.setattr(devices, "list_v4l2_devices", lambda: [devices.VideoDevice(0, "/dev/video0", "USB Camera", "capture")])
    assert devices.find_loopback_device() is None
