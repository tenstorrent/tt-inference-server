# VBench-I2V image–caption pairs for the image-to-video eval

Fifteen pairs from the image suite of VBench-I2V (part of VBench++,
https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v), used by the
Wan2.2 I2V eval when `I2V_EVAL_PAIRS=vbench_i2v`. Each image conditions a video
and its VBench caption is the prompt. `manifest.json` lists them in eval order.

- Picked to cover all eleven VBench-I2V categories, weighted toward people, and
  mostly captions that describe motion. Two first picks (VBench's horses and
  dandelion) are no longer downloadable from Pexels, so a walking tiger and
  reeds in the wind stand in for them.
- Four background scenes use VBench's camera-motion prompts (the caption plus
  ", camera pans left" and so on); `camera_motion` in the manifest names the
  move, which VBench's `camera_motion` dimension checks.
- Images: each entry's Pexels original (`source_url`), cropped with VBench's own
  16:9 box for that image (`diff_ratio_crop["16-9"]` in VBench's
  `i2v-bench-info.json`) and resized to 1280x720, the canvas Wan2.2 generates
  at. The originals matched VBench's recorded sizes, so the crops are VBench's.

The photos are from Pexels and are used under the Pexels License
(https://www.pexels.com/license/); VBench's captions and crop boxes are from the
VBench repository (Apache-2.0).

A GPU reference for this eval must use these exact files (not re-downloaded or
re-cropped), the captions in `manifest.json`, and the same seed, step count and
resolution as the run it is compared with.
