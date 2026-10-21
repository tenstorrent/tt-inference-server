# COCO image–caption pairs for the image-to-video eval

Five pairs from the COCO 2014 validation set, used by the Wan2.2 I2V eval: each
image conditions a video and its own caption is the prompt, so CLIP scores the
video against text that describes the image. `manifest.json` lists them in eval
order.

- Captions: the MLCommons COCO 2014 caption set the T2V eval already uses
  (`captions_source.tsv`, see `utils/sdxl_accuracy_utils/sdxl_accuracy_utils.py`).
- Images: downloaded from each entry's `coco_url` and center-cropped to 16:9, the
  aspect Wan2.2 generates at, so the server's resize does not squash them.
- Picked by hand from landscape images (width/height 1.3–1.8) for a subject that
  can move and no borders, watermarks or signatures.

COCO annotations are licensed CC BY 4.0. The images belong to their Flickr
owners under the licenses recorded per `coco_image_id` in COCO's 2014
annotations, and are used under the COCO Terms of Use
(https://cocodataset.org/#termsofuse).

A GPU reference for this eval must use these exact files (not re-downloaded or
re-cropped), the captions in `manifest.json`, and the same seed, step count and
resolution as the run it is compared with.
