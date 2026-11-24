"""Compare outputs from fresh processes running the standalone logits PoC."""
import argparse
import json
import math
import struct
from pathlib import Path


def read(path):
    data = json.loads(path.read_text())
    if not data['rounds'] or not data['inputs']:
        raise ValueError(f'{path}: no inference results')
    for row in data['rounds']:
        if len(row['jobs']) != data['contexts']:
            raise ValueError(f'{path}: context count mismatch')
        for job in row['jobs']:
            vector = job['embedding']
            if len(vector) != data['embedding_width'] or not all(math.isfinite(v) for v in vector):
                raise ValueError(f'{path}: invalid embedding')
            if not any(v != 0 for v in vector):
                raise ValueError(f'{path}: zero embedding')
    return data


def bits(vector):
    return struct.pack('<' + 'f' * len(vector), *vector)


def compare(baseline, candidate):
    for key in ['native_revision', 'architecture', 'device', 'contexts', 'pooling_type',
                'embedding_width', 'inputs', 'iterations', 'warmup']:
        if baseline[key] != candidate[key]:
            raise ValueError(f'incompatible {key}')
    if len(baseline['rounds']) != len(candidate['rounds']):
        raise ValueError('round count mismatch')
    checked = 0
    for left, right in zip(baseline['rounds'], candidate['rounds']):
        for key in ['warmup', 'iteration', 'input_index', 'tokens_per_job']:
            if left[key] != right[key]:
                raise ValueError(f'round metadata mismatch: {key}')
        for a, b in zip(left['jobs'], right['jobs']):
            if a['context_index'] != b['context_index'] or bits(a['embedding']) != bits(b['embedding']):
                delta = max(abs(x - y) for x, y in zip(a['embedding'], b['embedding']))
                raise ValueError(f'embedding differs: max_abs={delta}')
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidates', type=Path, nargs='+')
    args = parser.parse_args()
    baseline = read(args.baseline)
    if baseline['mode'] != 'baseline':
        raise ValueError('the reference result must use baseline mode')
    for path in args.candidates:
        candidate = read(path)
        count = compare(baseline, candidate)
        print(json.dumps({'mode': candidate['mode'], 'vector_pairs': count, 'bitwise_equal': True}))


if __name__ == '__main__':
    main()
