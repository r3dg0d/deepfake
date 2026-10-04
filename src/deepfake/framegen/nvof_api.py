"""ctypes bindings for NVIDIA Optical Flow (NVOFA) via libnvidia-opticalflow.

Headers vendored from https://github.com/NVIDIA/NVIDIAOpticalFlowSDK (BSD-3).
"""

from __future__ import annotations

import ctypes
from ctypes import CFUNCTYPE, POINTER, c_int, c_size_t, c_uint32, c_ulonglong, c_void_p

NV_OF_API_VERSION = (2 << 4) | 0  # matches vendored headers; driver also accepts newer

NV_OF_SUCCESS = 0
NV_OF_MODE_OPTICALFLOW = 1
NV_OF_PERF_LEVEL_SLOW = 5
NV_OF_PERF_LEVEL_MEDIUM = 10
NV_OF_PERF_LEVEL_FAST = 20
NV_OF_OUTPUT_VECTOR_GRID_SIZE_1 = 1
NV_OF_OUTPUT_VECTOR_GRID_SIZE_2 = 2
NV_OF_OUTPUT_VECTOR_GRID_SIZE_4 = 4
NV_OF_BUFFER_USAGE_INPUT = 1
NV_OF_BUFFER_USAGE_OUTPUT = 2
NV_OF_BUFFER_FORMAT_GRAYSCALE8 = 1
NV_OF_BUFFER_FORMAT_SHORT2 = 5
NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR = 2


class INIT_PARAMS(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("outGridSize", c_uint32),
        ("hintGridSize", c_uint32),
        ("mode", c_uint32),
        ("perfLevel", c_uint32),
        ("enableExternalHints", c_uint32),
        ("enableOutputCost", c_uint32),
        ("hPrivData", c_void_p),
        ("disparityRange", c_uint32),
        ("enableRoi", c_uint32),
    ]


class BUFFER_DESCRIPTOR(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("width", c_uint32),
        ("height", c_uint32),
        ("bufferUsage", c_uint32),
        ("bufferFormat", c_uint32),
    ]


class BUFFER_STRIDE(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("strideXInBytes", c_uint32), ("strideYInBytes", c_uint32)]


class CUDA_BUFFER_STRIDE_INFO(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("strideInfo", BUFFER_STRIDE * 3), ("numPlanes", c_uint32)]


class EXECUTE_INPUT_PARAMS(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("inputFrame", c_void_p),
        ("referenceFrame", c_void_p),
        ("externalHints", c_void_p),
        ("disableTemporalHints", c_uint32),
        ("padding", c_uint32),
        ("hPrivData", c_void_p),
        ("padding2", c_uint32),
        ("numRois", c_uint32),
        ("roiData", c_void_p),
    ]


class EXECUTE_OUTPUT_PARAMS(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("outputBuffer", c_void_p),
        ("outputCostBuffer", c_void_p),
        ("hPrivData", c_void_p),
    ]


class CUDA_API_FUNCTION_LIST(ctypes.Structure):
    _fields_ = [
        ("nvCreateOpticalFlowCuda", c_void_p),
        ("nvOFInit", c_void_p),
        ("nvOFCreateGPUBufferCuda", c_void_p),
        ("nvOFGPUBufferGetCUarray", c_void_p),
        ("nvOFGPUBufferGetCUdeviceptr", c_void_p),
        ("nvOFGPUBufferGetStrideInfo", c_void_p),
        ("nvOFSetIOCudaStreams", c_void_p),
        ("nvOFExecute", c_void_p),
        ("nvOFDestroyGPUBufferCuda", c_void_p),
        ("nvOFDestroy", c_void_p),
        ("nvOFGetLastError", c_void_p),
        ("nvOFGetCaps", c_void_p),
    ]


def _status(rc: int, what: str) -> None:
    if rc != NV_OF_SUCCESS:
        raise RuntimeError(f"{what} failed with NV_OF status {rc}")


class NvOFSession:
    """One Optical Flow session bound to the current CUDA context."""

    def __init__(self, width: int, height: int, *, perf_level: int = NV_OF_PERF_LEVEL_FAST, grid: int = 4) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for NVIDIA Optical Flow")
        torch.cuda.init()
        torch.zeros(1, device="cuda")  # ensure context

        self.width, self.height = int(width), int(height)
        self.grid = int(grid)
        self.out_w = (self.width + self.grid - 1) // self.grid
        self.out_h = (self.height + self.grid - 1) // self.grid

        self._cuda = ctypes.CDLL("libcuda.so.1")
        self._of = ctypes.CDLL("libnvidia-opticalflow.so.1")
        self._ctx = c_void_p()
        self._cuda.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        self._cuda.cuCtxGetCurrent.restype = c_int
        _status(self._cuda.cuCtxGetCurrent(ctypes.byref(self._ctx)), "cuCtxGetCurrent")
        if not self._ctx:
            raise RuntimeError("no CUDA context")

        fl = CUDA_API_FUNCTION_LIST()
        self._of.NvOFAPICreateInstanceCuda.argtypes = [c_uint32, POINTER(CUDA_API_FUNCTION_LIST)]
        self._of.NvOFAPICreateInstanceCuda.restype = c_int
        _status(self._of.NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, ctypes.byref(fl)), "NvOFAPICreateInstanceCuda")

        self._Create = CFUNCTYPE(c_int, c_void_p, POINTER(c_void_p))(fl.nvCreateOpticalFlowCuda)
        self._Init = CFUNCTYPE(c_int, c_void_p, POINTER(INIT_PARAMS))(fl.nvOFInit)
        self._CreateBuf = CFUNCTYPE(c_int, c_void_p, POINTER(BUFFER_DESCRIPTOR), c_uint32, POINTER(c_void_p))(
            fl.nvOFCreateGPUBufferCuda
        )
        self._GetPtr = CFUNCTYPE(c_ulonglong, c_void_p)(fl.nvOFGPUBufferGetCUdeviceptr)
        self._GetStride = CFUNCTYPE(c_int, c_void_p, POINTER(CUDA_BUFFER_STRIDE_INFO))(fl.nvOFGPUBufferGetStrideInfo)
        self._Execute = CFUNCTYPE(c_int, c_void_p, POINTER(EXECUTE_INPUT_PARAMS), POINTER(EXECUTE_OUTPUT_PARAMS))(
            fl.nvOFExecute
        )
        self._DestroyBuf = CFUNCTYPE(c_int, c_void_p)(fl.nvOFDestroyGPUBufferCuda)
        self._Destroy = CFUNCTYPE(c_int, c_void_p)(fl.nvOFDestroy)

        self._h = c_void_p()
        _status(self._Create(self._ctx, ctypes.byref(self._h)), "nvCreateOpticalFlowCuda")
        init = INIT_PARAMS(
            width=self.width,
            height=self.height,
            outGridSize=self.grid,
            hintGridSize=0,
            mode=NV_OF_MODE_OPTICALFLOW,
            perfLevel=perf_level,
            enableExternalHints=0,
            enableOutputCost=0,
            hPrivData=None,
            disparityRange=0,
            enableRoi=0,
        )
        _status(self._Init(self._h, ctypes.byref(init)), "nvOFInit")

        self._h_in = self._mkbuf(NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, self.width, self.height)
        self._h_ref = self._mkbuf(NV_OF_BUFFER_USAGE_INPUT, NV_OF_BUFFER_FORMAT_GRAYSCALE8, self.width, self.height)
        self._h_out = self._mkbuf(NV_OF_BUFFER_USAGE_OUTPUT, NV_OF_BUFFER_FORMAT_SHORT2, self.out_w, self.out_h)
        self._ptr_in = self._GetPtr(self._h_in)
        self._ptr_ref = self._GetPtr(self._h_ref)
        self._ptr_out = self._GetPtr(self._h_out)
        si = CUDA_BUFFER_STRIDE_INFO()
        _status(self._GetStride(self._h_in, ctypes.byref(si)), "stride in")
        self._in_pitch = si.strideInfo[0].strideXInBytes
        so = CUDA_BUFFER_STRIDE_INFO()
        _status(self._GetStride(self._h_out, ctypes.byref(so)), "stride out")
        self._out_pitch = so.strideInfo[0].strideXInBytes

        self._cuda.cuMemcpyHtoD_v2.argtypes = [c_ulonglong, c_void_p, c_size_t]
        self._cuda.cuMemcpyHtoD_v2.restype = c_int
        self._cuda.cuMemcpyDtoH_v2.argtypes = [c_void_p, c_ulonglong, c_size_t]
        self._cuda.cuMemcpyDtoH_v2.restype = c_int
        self._cuda.cuMemcpyDtoD_v2.argtypes = [c_ulonglong, c_ulonglong, c_size_t]
        self._cuda.cuMemcpyDtoD_v2.restype = c_int

        self._closed = False

    def _mkbuf(self, usage: int, fmt: int, w: int, h: int) -> c_void_p:
        d = BUFFER_DESCRIPTOR(width=w, height=h, bufferUsage=usage, bufferFormat=fmt)
        handle = c_void_p()
        _status(
            self._CreateBuf(self._h, ctypes.byref(d), NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, ctypes.byref(handle)),
            "nvOFCreateGPUBufferCuda",
        )
        return handle

    def _upload_gray(self, ptr: int, gray) -> None:
        import numpy as np

        g = np.ascontiguousarray(gray)
        if g.shape != (self.height, self.width):
            raise ValueError(f"gray shape {g.shape} != {(self.height, self.width)}")
        w = self.width
        pitch = self._in_pitch
        if pitch == w:
            _status(self._cuda.cuMemcpyHtoD_v2(ptr, g.ctypes.data, w * self.height), "HtoD")
        else:
            for y in range(self.height):
                _status(self._cuda.cuMemcpyHtoD_v2(ptr + y * pitch, g[y].ctypes.data, w), "HtoD row")


    def upload_gray_device(self, ptr: int, gray_dev_ptr: int, width: int | None = None) -> None:
        """Copy contiguous device gray (HxW uint8, row-major, pitch==width) into OF buffer."""
        w = width or self.width
        pitch = self._in_pitch
        if pitch == w:
            _status(self._cuda.cuMemcpyDtoD_v2(ptr, gray_dev_ptr, w * self.height), "DtoD")
        else:
            for y in range(self.height):
                _status(
                    self._cuda.cuMemcpyDtoD_v2(ptr + y * pitch, gray_dev_ptr + y * w, w),
                    "DtoD row",
                )

    def estimate_device(self, gray0_ptr: int, gray1_ptr: int, *, disable_temporal: bool = False):
        """Like estimate() but gray frames already live on device (contiguous HxW uint8)."""
        import numpy as np

        self.upload_gray_device(self._ptr_in, gray0_ptr)
        self.upload_gray_device(self._ptr_ref, gray1_ptr)
        ein = EXECUTE_INPUT_PARAMS(
            inputFrame=self._h_in,
            referenceFrame=self._h_ref,
            externalHints=None,
            disableTemporalHints=1 if disable_temporal else 0,
            padding=0,
            hPrivData=None,
            padding2=0,
            numRois=0,
            roiData=None,
        )
        eout = EXECUTE_OUTPUT_PARAMS(outputBuffer=self._h_out, outputCostBuffer=None, hPrivData=None)
        _status(self._Execute(self._h, ctypes.byref(ein), ctypes.byref(eout)), "nvOFExecute")

        raw = np.zeros((self.out_h, self._out_pitch), dtype=np.uint8)
        for y in range(self.out_h):
            _status(
                self._cuda.cuMemcpyDtoH_v2(raw[y].ctypes.data, self._ptr_out + y * self._out_pitch, self.out_w * 4),
                "DtoH",
            )
        vec = raw[:, : self.out_w * 4].view(np.int16).reshape(self.out_h, self.out_w, 2).astype(np.float32) / 32.0
        if self.grid == 1 and vec.shape[0] == self.height and vec.shape[1] == self.width:
            return vec
        # bilinear upsample of flow vectors (sharper warps than nearest-neighbor blocks)
        import cv2
        flow = cv2.resize(vec, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return flow

    def estimate(self, gray0, gray1, *, disable_temporal: bool = False):
        """Return float32 flow (H, W, 2) in pixels from gray0 → gray1 (upsampled from grid)."""
        import numpy as np

        self._upload_gray(self._ptr_in, gray0)
        self._upload_gray(self._ptr_ref, gray1)
        ein = EXECUTE_INPUT_PARAMS(
            inputFrame=self._h_in,
            referenceFrame=self._h_ref,
            externalHints=None,
            disableTemporalHints=1 if disable_temporal else 0,
            padding=0,
            hPrivData=None,
            padding2=0,
            numRois=0,
            roiData=None,
        )
        eout = EXECUTE_OUTPUT_PARAMS(outputBuffer=self._h_out, outputCostBuffer=None, hPrivData=None)
        _status(self._Execute(self._h, ctypes.byref(ein), ctypes.byref(eout)), "nvOFExecute")

        raw = np.zeros((self.out_h, self._out_pitch), dtype=np.uint8)
        for y in range(self.out_h):
            _status(
                self._cuda.cuMemcpyDtoH_v2(raw[y].ctypes.data, self._ptr_out + y * self._out_pitch, self.out_w * 4),
                "DtoH",
            )
        vec = raw[:, : self.out_w * 4].view(np.int16).reshape(self.out_h, self.out_w, 2).astype(np.float32) / 32.0
        if self.grid == 1 and vec.shape[0] == self.height and vec.shape[1] == self.width:
            return vec
        # bilinear upsample of flow vectors (sharper warps than nearest-neighbor blocks)
        import cv2
        flow = cv2.resize(vec, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return flow

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for attr in ("_h_in", "_h_ref", "_h_out"):
            h = getattr(self, attr, None)
            if h:
                try:
                    self._DestroyBuf(h)
                except Exception:
                    pass
                setattr(self, attr, None)
        h = getattr(self, "_h", None)
        if h:
            try:
                self._Destroy(h)
            except Exception:
                pass
            self._h = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def optical_flow_available() -> tuple[bool, str]:
    try:
        lib = ctypes.CDLL("libnvidia-opticalflow.so.1")
        ver = c_uint32(0)
        lib.NvOFGetMaxSupportedApiVersion.argtypes = [POINTER(c_uint32)]
        lib.NvOFGetMaxSupportedApiVersion.restype = c_int
        rc = lib.NvOFGetMaxSupportedApiVersion(ctypes.byref(ver))
        if rc != 0:
            return False, f"NvOFGetMaxSupportedApiVersion rc={rc}"
        return True, f"libnvidia-opticalflow API 0x{ver.value:x}"
    except OSError as e:
        return False, str(e)
