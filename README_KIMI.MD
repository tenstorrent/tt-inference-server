## Patches on tt-inference-server
This branch was created to patch remote execution of TT evals, benchmarks and tests on tenstorrent's console for Kimi-K2.6.


## How to run evals.
First create a venv and install pyyaml.

Then run the following command (gpu is just a placeholder until we have a more appropriate device name).
To run only one eval at a time you can comment/ uncomment evals lines 145-312 at [evals/eval_config.py](evals/eval_config.py). 

You will to obtain a huggingface token. For GPQA-Diamond you will need to grant access to the dataset on huggingface.

You can obtain an API_KEY directly at [https://console.tenstorrent.com/dashboard/inference/keys](https://console.tenstorrent.com/dashboard/inference/keyshttps://console.tenstorrent.com/dashboard/inference/keys)
```bash
HF_TOKEN="" API_KEY="" .venv/bin/python run.py --model Kimi-K2.6 --workflow evals --device gpu --server-url https://console.tenstorrent.com:443
```


## How to run tests
Model tests, can be run via this command.

```bash
HF_TOKEN="" API_KEY="" .venv/bin/python run.py --model Kimi-K2.6 --workflow tests --device gpu --server-url https://console.tenstorrent.com:443
```
