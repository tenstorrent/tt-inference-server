# Release process

The release process can be run locally on a laptop or on a remote server. 

## requirements
requirement is:
- Download only
    - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) (PAT)
        - Read access to tt-shield repo.
- Full release:
    - [GitHub Personal Access Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens) (PAT)
        - Read access to tt-shield repo.
        - Write access to tt-inference-server packages
    - crane CLI (https://github.com/google/go-containerregistry/tree/main/cmd/crane)

```bash
export GH_ID=tstescoTT
export GH_PAT=ghp_xxxxxxx
crane auth login ghcr.io -u ${GH_ID} -p ${GH_PAT}
# optionally login with docker CLI (if you only want to download logs and not do full release using crane)
docker login ghcr.io -u ${GH_ID} -p ${GH_PAT}
```

## Pre-release to `dev`

Make pre-release branch:
```bash
git branch -b pre-release-vx.x.0
```


## Step 1: parse Models CI run data

```bash
python3 scripts/release/models_ci_reader.py --max-runs 90
```

#### outputs
The outputs have the Models CI run numbers to demark the span of release
- `release_logs/models_ci_all_results_190_to_293.json`: this has full Models CI parsed data for analysis
- `release_logs/models_ci_last_good_190_to_293.json`: this is used downstream for release process

## step 2: update model_spec.py


```bash
python3 scripts/release/update_model_spec.py release_logs/models_ci_last_good_190_to_293.json
```

### step 2b: [if manual models] manual release model changes to model_spec.py

After changes are added, re-generate the Model Support `README.md` table and `model_specs_output.json` run:

```bash
python3 scripts/release/update_model_spec.py --output-only
```

#### outputs

- `workflows/model_spec.py`: diff has updates from Models CI most recent passing runs
- `model_specs_output.json`: all model specs fully expanded from the ModelSpecTemplates in `workflows/model_spec.py`
- `release_logs/release_models_diff.md`: summary of diff with links to specifci Models CI runs
- `README.md`: updates to the `Model Support` section

## step 3: generate pre-release artifacts

Promote Docker images from Models CI on GHCR from tt-shield repo to `release` images on tt-inference-server repo. 

For example, from:
- src: ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.5-ef93cf18b3aee66cc9ec703423de0ad3c6fde844-1d799da-52729064622
- dst: ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.5-ef93cf1-1d799da

```bash
python3 scripts/release/make_release_image_artifacts.py release_logs/models_ci_last_good_190_to_293.json --increment minor --dev
```

usage:
* positional: models_ci_last_good_json file e.g. release_logs/models_ci_last_good_190_to_293.json
* --increment {`major`, `minor`, `patch`}: this increments VERSION file before running
* --dev: pre-release setting to update `-dev-` images
* --release: targets `-release-` images
* --dry-run see what would happen without copying Docker images from tt-shield

#### outputs

- `release_logs/dev_artifacts_summary.md`: summary of Docker image changes
- `release_logs/dev_artifacts_summary.json`: JSON version of Docker image changes

### step 3b: [if manual models] build any manually added Model Spec Docker images

Start by promoting Models CI images if existing for manual models (e.g. if ad hoc or dispatch  CI job was used).
```bash
crane copy <src> <dst>
# e.g.
# crane copy https://ghcr.io/tenstorrent/tt-shield/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.0.5-f8f27288d6da50c0ac7fe8afce3c7e6db3b5f27f-91dddb0-52470823821 https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.1.0-f8f2728-91dddb0
```

Only if needed, will see in `release_logs/release_artifacts_summary.md` if any images need to be built.
This will build all missing `dev` containers for the given `model_spec.py` and push them:
```bash
python3 scripts/build_docker_images.py --push
```

#### outputs

- `workflow_logs/docker_build_logs/build_summary_{date}_{time}.json`: list of all outcomes of docker image script, includes `build_attempted`, `build_succeeded`, and `remote_exists` which list the status of docker images.
- `workflow_logs/docker_build_logs/build_{date}_{time}_{image-tag}.log`: run log + docker build log for specific image


## step 4: create pre-release PR

* Open tt-inference-server PR `pre-release-vx.x.0` to dev https://github.com/tenstorrent/tt-inference-server/compare/dev...
* manually inspect and review `model_spec.py` changes
* include: `release_logs/release_models_diff.md`
* include: `release_logs/dev_artifacts_summary.md`
* any manual changes from the automated edits should be noted

## Release to `main`

Once pre-release PR is merged being release to `main`. Following git workflow in docs/development.md make RC branch 

Make pre-release branch:
```bash
git branch -b rc-vx.x.0
```

## step 1: generate release artifacts

Promote Docker images from Models CI on GHCR from tt-shield repo to `release` images on tt-inference-server repo. 

```bash
python3 scripts/release/make_release_image_artifacts.py release_logs/models_ci_last_good_190_to_293.json --release
```

#### outputs

- `release_logs/release_artifacts_summary.md`: summary of Docker image changes
- `release_logs/release_artifacts_summary.json`: JSON version of Docker image changes

### step 1b: [if manual models] build any manually added Model Spec Docker images

Only if needed, will see in `release_logs/release_artifacts_summary.md` if any images need to be built.
```bash
python3 scripts/build_docker_images.py --push --release
```

#### outputs

- `workflow_logs/docker_build_logs/build_summary_{date}_{time}.json`: list of all outcomes of docker image script, includes `build_attempted`, `build_succeeded`, and `remote_exists` which list the status of docker images.
- `workflow_logs/docker_build_logs/build_{date}_{time}_{image-tag}.log`: run log + docker build log for specific image

## step 2: create release PR

* Open tt-inference-server PR `rc-vx.x.0` to `main` https://github.com/tenstorrent/tt-inference-server/compare/main...
* manually inspect and review all changes
* if possible: run `python3 scripts/release/update_model_spec.py` to generate the `release_logs/release_models_diff.md` against the `main` model_spec.py.
* include: `release_logs/release_artifacts_summary.md`
* include: `release_logs/release_models_diff.md`

## Release notes:

* must be added describing new supported vLLM features 
* add notes for changes to model support and performance (if possible use `release_logs/release_models_diff.md`)
