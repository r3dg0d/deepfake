# Synthetic validation media

`fullbody-source.png` and `fullbody-reference.png` were generated with the
built-in imagegen tool on 2026-10-21, specifically for this project's validation.
They depict fictional adults and do not use uploaded photographs or real-person
references. The images are local test references, not evidence of model output.
These project assets follow the repository's MIT terms; no model weights are
included. Neural before/after outputs must carry the existing disclosure mark.

Prompt set (built-in mode):

1. Photorealistic fictional adult with short dark hair, plain gray athletic top,
   charcoal trousers and white trainers; full body, head/hands/feet visible,
   relaxed front-facing A-pose, softly lit neutral studio, natural anatomy,
   no brands/text/props/other people; invented computer-vision validation fixture.
2. Photorealistic fictional adult woman with short auburn hair, vivid teal
   technical flight jumpsuit with silver shoulder/knee panels and dark boots;
   full body, head/hands/feet visible, relaxed front-facing A-pose, softly lit
   neutral studio, natural anatomy, no brands/text/props/other people;
   invented neural character reference.

`scripts/prepare-body-fixture.py` encodes a 21-frame, 16 FPS translated-still
driving clip from the first image. It does not show walking, independently
moving joints or expressions. Never present it as a motion-quality benchmark.
The original generated PNGs are retained unchanged.
