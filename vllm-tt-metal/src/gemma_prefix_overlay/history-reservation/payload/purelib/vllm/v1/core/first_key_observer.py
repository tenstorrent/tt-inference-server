"""Opt-in first-full-key observation. No cache policy or tensor operations."""
import hashlib
import json
import os
import time
from collections import Counter
from functools import wraps

def observational(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if self.failed:
            return None
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            self.failed = type(exc).__name__
            self.request = None
            self.live.clear()
            self.keys.clear()
            try:
                self.sink(dict(kind="coverage_lost", sequence=self.sequence, reason=self.failed))
            except Exception:
                pass
            return None
    return guarded



class FirstKeyObserver:
    def __init__(self, pool, sink=None, key_limit=4096, event_limit=20000):
        self.failed = None
        self.pool = pool
        self.sink = sink or (lambda row: print('GEMMA_FIRST_KEY', json.dumps(row), flush=True))
        self.key_limit, self.event_limit = key_limit, event_limit
        self.secret = os.urandom(32)  # Never logged; fingerprints only join within this engine.
        self.keys = set()
        self.live = {}  # block_id -> tracked first keys; not cache ownership.
        self.request = None
        self.counts = Counter()
        self.sequence = 0
        self.dropped = 0
        self.emit('begin', hash_block_size=pool.hash_block_size, full_group=5,
                  key_limit=key_limit, event_limit=event_limit)

    def fingerprint(self, value):
        if not isinstance(value, bytes):
            value = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        return hashlib.blake2b(value, key=self.secret, digest_size=16).hexdigest()

    @observational
    def emit(self, kind, **fields):
        self.counts[kind] += 1
        self.sequence += 1
        if self.sequence > self.event_limit:
            self.dropped += 1
            self.failed = 'event_limit'
            self.request = None
            self.live.clear()
            self.keys.clear()
            self.sink(dict(kind='coverage_exhausted', sequence=self.sequence,
                           reason='event_limit', counts=dict(self.counts)))
            return
        self.sink(dict(kind=kind, sequence=self.sequence, mono_ns=time.monotonic_ns(),
                       free_blocks=self.pool.get_num_free_blocks(),
                       hash_entries=len(self.pool.cached_block_hash_to_block), **fields))

    def track(self, key):
        if key in self.keys:
            return True
        if len(self.keys) >= self.key_limit:
            self.emit('key_untracked', reason='key_limit')
            return False
        self.keys.add(key)
        return True

    @observational
    def begin_request(self, request, enabled):
        # This diagnostic is intentionally confined to the accepted 128-token full group.
        supported = self.pool.hash_block_size == 128
        hashes = request.block_hashes
        key = hashes[0] + (5).to_bytes(4, 'big') if hashes and supported else None
        tracked = key is not None and self.track(key)
        self.request = (request.request_id, key) if tracked else None
        ids = request.all_token_ids[:128]
        block = self.pool.cached_block_hash_to_block.get_one_block(key) if tracked else None
        if block is not None:
            self.live.setdefault(block.block_id, set()).add(key)
        self.emit('request', request_id=request.request_id, first_index=0, group=5,
                  key=self.fingerprint(key) if key is not None else None,
                  tokens_fingerprint=self.fingerprint(ids), first_token_count=len(ids),
                  salt_present=bool(request.cache_salt),
                  salt_fingerprint=self.fingerprint(request.cache_salt),
                  lora_present=request.lora_request is not None,
                  mm_present=bool(request.mm_features),
                  embeds_present=request.prompt_embeds is not None,
                  hash_count=len(hashes), prompt_tokens=request.num_tokens,
                  max_length=request.num_tokens-1, lookup_enabled=enabled,
                  supported_geometry=supported, tracked=tracked,
                  probe_block=None if block is None else block.block_id,
                  probe_refcount=None if block is None else block.ref_cnt,
                  counts=dict(self.counts), dropped=self.dropped)

    @observational
    def registration_input(self, request, blocks, cached, full, block_size, group, mask):
        if group != 5 or cached != 0:
            return
        self.emit('registration_input', request_id=request.request_id, group=group,
                  num_cached=cached, num_full=full, block_size=block_size,
                  hash_count=len(request.block_hashes),
                  first_null=blocks[0].is_null if blocks else None,
                  first_mask=None if mask is None or not mask else mask[0],
                  first_key=self.fingerprint(request.block_hashes[0] + (5).to_bytes(4, 'big'))
                  if request.block_hashes and self.pool.hash_block_size == 128 else None)

    @observational
    def end_request(self):
        self.emit('request_end', request_id=self.request[0] if self.request else None, counts=dict(self.counts), dropped=self.dropped)
        self.request = None

    @observational
    def lookup(self, key, block):
        if self.request is None or key != self.request[1]:
            return
        self.emit('lookup', request_id=self.request[0], key=self.fingerprint(key),
                  group=5, index=0, hit=block is not None,
                  block_id=None if block is None else block.block_id,
                  refcount=None if block is None else block.ref_cnt)

    @observational
    def insert(self, key, block, num_tokens):
        if key[-4:] != (5).to_bytes(4, 'big') or (num_tokens != 128 and key not in self.keys):
            return
        if not self.track(key):
            return
        self.live.setdefault(block.block_id, set()).add(key)
        self.emit('register', key=self.fingerprint(key), group=5, block_id=block.block_id,
                  refcount=block.ref_cnt, num_tokens=num_tokens)

    @observational
    def remove(self, keys, block, reason):
        for key in keys:
            if key not in self.keys:
                continue
            replacement = self.pool.cached_block_hash_to_block.get_one_block(key)
            self.emit('remove', key=self.fingerprint(key), block_id=block.block_id,
                      refcount=block.ref_cnt, reason=reason,
                      key_still_present=replacement is not None,
                      surviving_block=None if replacement is None else replacement.block_id)
            self.live.get(block.block_id, set()).discard(key)
        if not self.live.get(block.block_id):
            self.live.pop(block.block_id, None)

    @observational
    def ref(self, kind, block):
        keys = self.live.get(block.block_id)
        if keys:
            self.emit(kind, block_id=block.block_id, refcount=block.ref_cnt,
                      keys=sorted(self.fingerprint(x) for x in keys))

    @observational
    def reset(self):
        self.live.clear()
        self.emit('reset', reason='successful_prefix_reset')
