# MiniMax-H3 input requirements

## Modes


| Mode                                      | Inputs                                        |
| ----------------------------------------- | --------------------------------------------- |
| T2VA (text to video + audio)              | prompt                                        |
| FL2VA (first/last frame to video + audio) | prompt, first and/or last keyframe            |
| Ref2VA (reference to video + audio)       | prompt, reference images, videos and/or audio |


A deployment serves either FL2VA or Ref2VA. On an FL2VA deployment, a request with no keyframe runs as T2VA. Keyframes and references cannot be combined.

## Common parameters

These apply to every mode.


| Parameter         | Required | Requirement                                                                                                                                                                                | Default                                                     |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `prompt`          | Yes      | Non-empty; at most 3008 tokens                                                                                                                                                             | —                                                           |
| `duration`        | No       | Whole seconds, 4 to 15                                                                                                                                                                     | 5                                                           |
| `aspect_ratio`    | No       | One of `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16`. Ignored if `height` and `width` are set. In FL2VA, also ignored if a keyframe is sent. See [Output resolution](#output-resolution) | `16:9`                                                      |
| `height`, `width` | No       | Set both or neither. Each must be a multiple of 32, `height × width` must be at most 1,032,192 (768 × 1344), and `width : height` must be between 1:4 and 4:1. Any ratio in that range is accepted, not only the `aspect_ratio` values. These set the shape of the output, not its exact size; see [Output resolution](#output-resolution) | Computed from `aspect_ratio`, or in FL2VA from the keyframe |
| `seed`            | No       | Integer                                                                                                                                                                                    | 0                                                           |


### Output resolution

The output resolution is derived from one of the following, in this order:

- `height` and `width`, if set
- in FL2VA, the first keyframe, or the last keyframe if only the last is sent
- `aspect_ratio`

It is then resized:

- The short side is scaled to 768 pixels, up or down.
- If the area is then above about 1,032,192 pixels (768 × 1344), both sides are scaled down to fit.
- Each side is rounded to a multiple of 32, so the output aspect ratio can differ slightly from the
requested one, and the area can be slightly above the cap.

For example, `16:9` gives 1344 × 768, `21:9` gives 1536 × 672, and `height: 480, width: 832`
also gives 1344 × 768.

## FL2VA: keyframes


| Rule         | Requirement                                                                                                              |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Count        | 1 or 2 keyframes                                                                                                         |
| `image`      | Base64-encoded image, or an `http(s)` URL                                                                                |
| Format       | PNG, JPEG, WebP or other image formats supported by [PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html). The upload endpoint takes `image/png`, `image/jpeg` or `image/webp` only |
| File size    | ≤ 30 MB                                                                                                                  |
| Dimensions   | Each side 256 to 5760 px                                                                                                 |
| Aspect ratio | 1:4 to 4:1                                                                                                               |


Each keyframe is resized to the output resolution:

- The first frame is stretched to fit, so it may be distorted if its aspect ratio is different from the output's. If only the last frame is sent, the same applies to it.
- If both frames are sent, the last frame is resized to fill the output, keeping its aspect
ratio, and then center-cropped.

To avoid distortion or cropping, send keyframes with the same aspect ratio as the output.

## Ref2VA: references

Each reference is either base64-encoded or an `http(s)` URL.

### Counts


| Rule        | Requirement                                           |
| ----------- | ----------------------------------------------------- |
| Images      | ≤ 9                                                   |
| Videos      | ≤ 3                                                   |
| Audio clips | ≤ 3 ; Must be paired with at least one image or video |
| Total       | 1 to 12                                               |



### Per-type requirements


| Type  | Format                                                            | File size | Other                                                                    |
| ----- | ----------------------------------------------------------------- | --------- | ------------------------------------------------------------------------ |
| Image | PNG, JPEG, WebP or other image formats supported by [PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html)                    | ≤ 30 MB   | Aspect ratio 1:4 to 4:1                                                  |
| Video | MP4, MOV or other video formats supported by [FFmpeg](https://ffmpeg.org/ffmpeg-formats.html); must report a frame rate | ≤ 50 MB   | Aspect ratio 1:4 to 4:1; each clip 2 to 15 s; all videos combined ≤ 15 s |
| Audio | WAV, MP3 or other audio formats supported by [FFmpeg](https://ffmpeg.org/ffmpeg-formats.html); mono or stereo           | ≤ 15 MB   | Each clip 2 to 15 s; all audio clips combined ≤ 15 s                     |


A video's soundtrack, if it has one, is used as a reference too.

## Request size and URLs


| Rule              | Requirement                                                                                      |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| Inline media      | All base64 media in one request ≤ 64,000,000 characters (about 48 MB). Send larger assets by URL |
| URL download size | ≤ 50 MB per file; the per-type caps above still apply                                            |
| URL hosts         | Must be a host the deployment allows                                                             |
| URL download time | All URLs in one request must download within 300 s, with at most 5 redirects                     |


