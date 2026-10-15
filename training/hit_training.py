"""Training-only data filtering, hard replay and a size-aware box surrogate."""
from collections import defaultdict
import random

import torch


def remove_ambiguous_intents(targets):
    def key(target, text):
        return target['source_id'], ' '.join(text.casefold().split())
    destinations = defaultdict(set)
    for target in targets:
        for text in target['intents']:
            destinations[key(target, text)].add(tuple(target['target']))
    ambiguous = {k for k, points in destinations.items() if len(points) > 1}
    cleaned, removed = [], 0
    for target in targets:
        intents = [s for s in target['intents'] if key(target, s) not in ambiguous]
        removed += len(target['intents']) - len(intents)
        if intents:
            cleaned.append({**target, 'intents': intents})
    return cleaned, {'ambiguous_keys': len(ambiguous), 'removed_expressions': removed,
                     'removed_targets': len(targets) - len(cleaned)}


def box_penalty(expected, bbox_pixels, image_size):
    """Robust distance outside the inner 80% box, scaled by target size.

    expected has x/y in [0,1] image coordinates. This teacher-forced
    surrogate is not a direct loss on autoregressive click success.
    """
    scale = expected.new_tensor(image_size)
    box = expected.new_tensor(bbox_pixels).reshape(2, 2) / scale
    span = box[1] - box[0]
    if bool((span <= 0).any()):
        raise ValueError('Box must have positive width and height')
    inset = 0.1 * span
    distance = torch.relu(box[0] + inset - expected) + torch.relu(expected - box[1] + inset)
    # A 1% image-size floor avoids extreme gradients on tiny annotations.
    return torch.log1p(distance / span.clamp_min(0.01)).mean()


def mining_indices(targets, seed):
    """One target per training screen; alternate small-target and random picks."""
    grouped = defaultdict(list)
    for index, target in enumerate(targets):
        grouped[target['source_id']].append(index)
    rng = random.Random(seed)
    screens = sorted(grouped)
    rng.shuffle(screens)
    def short_side(index):
        target = targets[index]
        x0, y0, x1, y1 = target['bbox_pixels']
        w, h = target['image_size']
        return min((x1 - x0) / w, (y1 - y0) / h)
    selected = []
    for i, screen in enumerate(screens):
        candidates = grouped[screen]
        if i % 2 == 0:
            candidates = sorted(candidates, key=short_side)[:max(1, len(candidates) // 4)]
        selected.append(rng.choice(candidates))
    return selected


def selection_key(result, criterion):
    if criterion == 'bbox':
        return (-result['bbox_hits'], result['selection_error'])
    return (result['selection_error'],)


class ReplaySampler:
    def __init__(self, targets, hard_indices, seed):
        self.targets = targets
        self.rng = random.Random(seed)
        self.pools = {'general': list(range(len(targets))), 'hard': sorted(set(hard_indices))}
        if any(i < 0 or i >= len(targets) for i in self.pools['hard']):
            raise ValueError('Hard target index is outside training data')
        self.decks = {key: [] for key in self.pools}
        self.visits = [0] * len(targets)
        self.intents = [self.rng.sample(t['intents'], len(t['intents'])) for t in targets]
        self.counts = {'general': 0, 'hard': 0}

    def next(self, hard=False):
        pool = 'hard' if hard and self.pools['hard'] else 'general'
        if not self.decks[pool]:
            self.decks[pool] = self.pools[pool].copy()
            self.rng.shuffle(self.decks[pool])
        index = self.decks[pool].pop()
        instruction = self.intents[index][self.visits[index] % len(self.intents[index])]
        self.visits[index] += 1
        self.counts[pool] += 1
        return {**self.targets[index], 'instruction': instruction}
