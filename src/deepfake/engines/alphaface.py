from __future__ import annotations

from .base import EngineCapabilities, TransformationEngine


class AlphaFaceEngine(TransformationEngine):
    """Wrap the existing implementation without changing its inference path."""

    def load(self, config):
        from ..pipeline import FaceSwapPipeline

        self.pipeline = FaceSwapPipeline(config)

    def prepare_reference(self, reference):
        self.pipeline.set_source(reference)

    def process_frame(self, frame, context):
        return self.pipeline.swap_frame(frame)

    def capabilities(self):
        return EngineCapabilities(supports_face=True, supports_streaming=True, supports_cuda=True)

    def unload(self):
        self.pipeline.close()

    # Existing realtime and benchmark contracts.
    set_source = prepare_reference
    close = unload

    def swap_frame(self, frame):
        return self.pipeline.swap_frame(frame)
