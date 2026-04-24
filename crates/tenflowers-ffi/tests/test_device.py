"""Tests for PyDevice / PyDeviceKind Python bindings.

These tests document the expected Python API and will run once a
`tenflowers` wheel is installed in the environment.
"""


def test_device_cpu_repr():
    from tenflowers import Device
    d = Device.cpu()
    assert repr(d) == "Device.cpu()"
    assert str(d) == "cpu"


def test_device_cpu_is_cpu():
    from tenflowers import Device
    d = Device.cpu()
    assert d.is_cpu
    assert not d.is_gpu
    assert not d.is_rocm
    assert d.device_id == 0


def test_device_gpu_repr():
    from tenflowers import Device
    d = Device.gpu(0)
    assert repr(d) == "Device.gpu(0)"
    assert str(d) == "gpu:0"
    assert "gpu(0)" in repr(d)
    assert d.is_gpu
    assert not d.is_cpu
    assert d.device_id == 0


def test_device_gpu_id_1():
    from tenflowers import Device
    d = Device.gpu(1)
    assert d.device_id == 1
    assert repr(d) == "Device.gpu(1)"


def test_device_rocm_repr():
    from tenflowers import Device
    d = Device.rocm(2)
    assert repr(d) == "Device.rocm(2)"
    assert str(d) == "rocm:2"
    assert d.is_rocm
    assert d.device_id == 2


def test_device_equality():
    from tenflowers import Device
    a = Device.cpu()
    b = Device.cpu()
    assert a == b

    c = Device.gpu(0)
    assert a != c


def test_device_from_string_cpu():
    from tenflowers import Device
    d = Device.from_string("cpu")
    assert d.is_cpu


def test_device_from_string_gpu():
    from tenflowers import Device
    d = Device.from_string("gpu:0")
    assert d.is_gpu
    assert d.device_id == 0


def test_device_tensor_device_property():
    """PyTensor.device should return the CPU device for all current tensors."""
    from tenflowers import Device, zeros
    t = zeros([2, 3])
    dev = t.device
    assert dev.is_cpu
    assert repr(dev) == "Device.cpu()"
