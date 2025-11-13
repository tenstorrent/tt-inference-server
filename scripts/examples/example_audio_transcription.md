# Example Audio Transcription

Start tt-media-server in docker container:
```bash
python3 run.py --model whisper-large-v3 --device t3k --workflow server --docker-server --dev-mode
```

Note: whisper model implementation is single chip. It scales onto multple chips but requires 1 device with PCIe access so for LoudBox/QuietBox (t3k) it can only run on 4 devices (the L chip on the n300 boards). A 2 chip implementation could be added to remove this scaling limitation for n300 and LoudBox/QuietBox. Galaxy does not have this limitation and each of the 32 galaxy modules can run whisper with a tt-media-server worker associated.

## non-streaming output
Example script downloads librispeech samples are uses HTTP API for audio transcription:
```bash
python3 scripts/examples/example_audio_request.py --samples 32 --concurrency 4
```

Output logs like:
```log
...
[non-stream] E2EL=5.63s | audio_time=5.12s | RTR=0.91x | text='but the bear, instead of obeying, maintained the seat it had taken and growled.'
[non-stream] E2EL=5.87s | audio_time=5.66s | RTR=0.96x | text='Then, as if satisfied of their safety, the scout left his position and slowly entered the place.'
[non-stream] E2EL=8.66s | audio_time=10.09s | RTR=1.17x | text='the cunning man is afraid that his breath will blow upon his brothers and take away their courage too continued david improving the hint he received they must stand further off'
[non-stream] E2EL=9.12s | audio_time=9.70s | RTR=1.06x | text='it was silent and gloomy being tenanted solely by the captive and lighted by the dying embers of a fire which had been used for the purpose of cookery'
[non-stream] E2EL=7.00s | audio_time=8.23s | RTR=1.18x | text='uncas occupied a distant corner in a reclining attitude being rigidly bound both hands and feet by strong and painful withes'
[non-stream] E2EL=8.24s | audio_time=8.89s | RTR=1.08x | text='the scout who had left david at the door to ascertain they were not observed thought it prudent to preserve his disguise until assured of their privacy'
[non-stream] E2EL=7.29s | audio_time=5.33s | RTR=0.73x | text='What shall we do with the mingos at the door? They count six, and this singer is as good as nothing.'
Completed: successes=100, failures=0, total=100 | inference_time=162.15s | audio_total=670.57s | FULL_RTR=4.14x
```

## Streaming output:


```bash
# with streaming
python3 scripts/examples/example_audio_request.py --samples 32 --concurrency 4 --stream
```

Output logs like:
```log
...
[stream] req=10 chunk=25 text='action'
[stream] req=10 chunk=26 text='the'
[stream] req=10 chunk=27 text='whole'
[stream] req=10 chunk=28 text='day'
[stream] req=10 chunk=29 text='.'
[stream] req=10 chunk=30 text='It is you who are mistaken, Ra oul. I have read his distress in his eyes, in his every gesture and action the whole day.'
[stream] E2EL=7.97s | TTFT=1.21s | audio_time=7.28s | RTR=0.91x | chunks=30
Completed: successes=10, failures=0, total=10 | inference_time=25.07s | audio_total=91.52s | FULL_RTR=3.65x
```