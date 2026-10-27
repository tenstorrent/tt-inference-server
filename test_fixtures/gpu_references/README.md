# GPU references

Clips from upstream diffusers on a GPU, for the same eval requests TT serves. The VBench I2V scorer (`test_module/_test_common/vbench_i2v.py`) compares TT's eval clips with them on every run. It reports each VBench dimension against the GPU scores, and each frame's PSNR to the GPU clip.

One folder per model, with one folder per pair set and resolution inside it:

- the clips (`NN_gpu.mp4`)
- `index.json`: each clip's prompt and conditioning image, in the layout the video eval saves
- `provenance.json`: GPU, library versions and generation settings
- `vbench.json`: the clips' VBench scores, from the same scorer

The scorer picks the folder whose `index.json` holds every request in the TT run, at the TT clips' resolution. Clips are matched by prompt and conditioning image, not by file name.

| Folder | Pairs | Resolution | Made with |
|---|---|---|---|
| `wan2.2_i2v_a14b/coco_i2v_720p` | 5 COCO pairs | 1280×720, 81 frames | `scripts/wan_i2v_gpu_reference.py --pairs coco_i2v --resolution 720p` |
| `wan2.2_i2v_a14b/vbench_i2v_720p` | 15 VBench-I2V pairs | 1280×720, 81 frames | `scripts/wan_i2v_gpu_reference.py --pairs vbench_i2v --resolution 720p` |

The TT pipeline and diffusers draw bit-identical initial noise from the same seed (torch's CPU generator, float32), so a TT clip and its reference start from the same latent. Frame 0 is typically 27–38 dB PSNR apart. Later frames drift to about 10–25 dB as the numerical differences between the platforms build up over the 40 steps. A drop well below that range points to a change in the TT pipeline, not to sampling.

To add a reference, run the script on a CUDA GPU and score the output with the scorer, using `--gpu-reference none`. Then copy the clips and the `index.json`, `provenance.json` and `vbench.json` files into a new folder here.
