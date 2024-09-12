def batch_top_pk_logits_efficient_multi_params(
    logits,
    top_ps=[0.9],
    top_ks=[10],
    temperatures=[1.0],
    return_probs=False,
    skip_token=11,
):
    """
    Handle top_p and top_k sampling when a given batch has different params.
    This is quite rare as few users send non-default top_p and top_k values.
    """
    out_tokens = []
    for b_logits, p, k, temperature in zip(logits, top_ps, top_ks, temperatures):
        if p is None or k is None:
            # skip None users
            token = torch.tensor([skip_token])
        else:
            token = batch_top_pk_logits_efficient_same_params(
                b_logits, p=p, k=k, temperature=temperature
            )

        out_tokens.append(token)
    return torch.concat(out_tokens)


def batch_top_pk_logits_efficient_same_params(logits, p=0.9, k=40, temperature=1.0):
    # do not keep the entire vocab size after top k. Instead, keep the k size tensor and record the associated indices
    top_k_values, top_k_indices = torch.topk(logits, k=k)
    # replace any nans with 0's
    top_k_values = torch.where(
        torch.isnan(top_k_values), torch.zeros_like(top_k_values), top_k_values
    )
    top_p_values = top_k_top_p_filtering(top_k_values, top_p=p)
    probs = F.softmax(top_p_values / temperature, dim=-1)
    top_k_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token = top_k_indices.gather(-1, top_k_id.unsqueeze(-1)).squeeze(-1)
    return token


def check_if_all_equal(top_ps, top_ks, temperatures):
    # Remove None values from the lists
    top_ps = [p for p in top_ps if p is not None]
    top_ks = [k for k in top_ks if k is not None]
    temperatures = [t for t in temperatures if t is not None]
    if not top_ps or not top_ks or not temperatures:
        return False
    # Check if all elements in the list are equal
    all_top_ps_equal = all(p == top_ps[0] for p in top_ps)
    all_top_ks_equal = all(k == top_ks[0] for k in top_ks)
    all_temperatures_equal = all(t == temperatures[0] for t in temperatures)
    return all_top_ps_equal and all_top_ks_equal and all_temperatures_equal


def first_non_none(seq):
    return next((x for x in seq if x is not None), None)


def batch_top_pk_logits_efficient(
    logits, top_ps=[0.9], top_ks=[40], temperatures=[1.0]
):
    if check_if_all_equal(top_ps, top_ks, temperatures):
        # logits seq_len dimension is removed
        return batch_top_pk_logits_efficient_same_params(
            logits[:, -1, :],
            p=first_non_none(top_ps),
            k=first_non_none(top_ks),
            temperature=first_non_none(temperatures),
        )
    else:
        return batch_top_pk_logits_efficient_multi_params(
            logits, top_ps=top_ps, top_ks=top_ks, temperatures=temperatures
        )
