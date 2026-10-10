# Gemma 4 31B SWE Verified 50-case selection

Source: [SWE-bench Verified test split at revision
`c104f840`](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/tree/c104f840cc67f8b6eec6f759ebc8b2693d585d4a),
500 instances.

Keep the first twenty IDs below in their existing order. Exclude them
from the 500-case split, sort remaining IDs, then use one
`random.Random(20261011)` instance to sample 11 `<15 min fix`, 17
`15 min - 1 hour`, and 2 `1-4 hours` IDs in that order. Append the sampled
IDs in the listed order. The 50-case difficulty counts are 19/27/4/0
for `<15 min`, `15 min–1 hour`, `1–4 hours`, and `>4 hours`; 22/50 cases
are Django. The full split has 194/261/42/3 difficulty counts and
231/500 Django cases.

## Existing twenty

- `django__django-11299`
- `astropy__astropy-14096`
- `matplotlib__matplotlib-25332`
- `sympy__sympy-13551`
- `scikit-learn__scikit-learn-14629`
- `django__django-11820`
- `psf__requests-2317`
- `pylint-dev__pylint-4970`
- `sphinx-doc__sphinx-8265`
- `sympy__sympy-16597`
- `django__django-14315`
- `django__django-13109`
- `django__django-15375`
- `django__django-16485`
- `django__django-11206`
- `django__django-15814`
- `django__django-13344`
- `sympy__sympy-19783`
- `sphinx-doc__sphinx-8638`
- `pytest-dev__pytest-7432`

## New thirty

- `pytest-dev__pytest-7982`
- `django__django-16899`
- `scikit-learn__scikit-learn-11310`
- `django__django-16642`
- `django__django-15572`
- `django__django-11239`
- `django__django-11119`
- `django__django-9296`
- `pytest-dev__pytest-6202`
- `django__django-11099`
- `sympy__sympy-18763`
- `astropy__astropy-13453`
- `scikit-learn__scikit-learn-10908`
- `sympy__sympy-14976`
- `sphinx-doc__sphinx-10466`
- `pytest-dev__pytest-5631`
- `django__django-14771`
- `django__django-13315`
- `sympy__sympy-22914`
- `pylint-dev__pylint-6528`
- `pydata__xarray-3677`
- `matplotlib__matplotlib-23412`
- `sympy__sympy-13974`
- `django__django-11815`
- `django__django-16819`
- `sympy__sympy-20438`
- `astropy__astropy-14365`
- `astropy__astropy-12907`
- `django__django-14011`
- `django__django-14631`

