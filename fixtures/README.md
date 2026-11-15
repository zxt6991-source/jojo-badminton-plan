# Synthetic contract fixture

`demo-analysis.json`, `demo-video.json`, and the two Float32 files are generated data. There is no associated recorded video. The demo is for exercising the player, transport schema, and confidence behavior; it is not evidence of pose estimation or detection accuracy.

Regenerate with `npx tsx fixtures/generate.ts`. The six-second sequence contains 720 captured frames at 120 FPS and three right-handed synthetic swings. The final swing deliberately reduces wrist/elbow confidence to demonstrate suppression. The second uses visual evidence only. No timing performance measurements are fabricated (`stageTimingsMs` is empty).

Binary storage is little-endian Float32: timestamps `[frame]` in capture seconds and keypoints `[frame][65][x,y,confidence]`. The cwb-65 order is COCO body 17, feet 6, left hand 21, right hand 21; the 68 face points are omitted.

A real validation set must contain independently labeled videos and consent records. These synthetic fixtures cannot establish the accuracy or throughput targets in the P0a plan.

The v0.2 demo also varies hip height, knee bend and projected torso lean to exercise whole-body highlighting and angular-speed plots. This motion is an illustrative animation, not a biomechanically validated technique or ideal sequence.
