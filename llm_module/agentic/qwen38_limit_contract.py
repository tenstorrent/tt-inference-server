"""Host-only candidate limit contract; does not select or launch a serving arm."""
import copy

def aligned_terminal_kwargs(kwargs, *, server_context_tokens, safety_tokens=8192):
    """Align diagnostic metadata to explicit API cap after conservative validation.

    Safety covers pending/template overhead, not a measured future prompt bound.
    Actual tokenized requests still need the server's native context validation.
    """
    result=copy.deepcopy(kwargs)
    cap=result['llm_kwargs']['max_tokens']
    inputs=result['model_info']['max_input_tokens']
    for value in (cap,inputs,server_context_tokens,safety_tokens):
        if type(value) is not int or value <= 0:
            raise ValueError('Positive integer context limits required')
    if cap not in (32768,65536):
        raise ValueError('Candidate requires release32768 or proposed65536 cap')
    if safety_tokens < 8192:
        raise ValueError('At least8192 tokens of template/pending safety required')
    if inputs+cap+safety_tokens > server_context_tokens:
        raise ValueError('Input plus output plus safety exceeds served context')
    result['model_info']['max_output_tokens']=cap
    return result
