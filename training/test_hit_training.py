"""CPU checks for hit-focused loss, leakage boundaries and replay behavior."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from hit_training import box_penalty, mining_indices, remove_ambiguous_intents, ReplaySampler, selection_key
from train_coordinates import coordinate_loss, prepare_design, ROOT


class HitTrainingTests(unittest.TestCase):
    def test_box_geometry_and_gradients(self):
        # Non-square screen: (100,50)-(300,150) maps to (.1,.1)-(.3,.3).
        bbox, size = [100, 50, 300, 150], [1000, 500]
        center = torch.tensor([.2, .2], requires_grad=True)
        self.assertEqual(box_penalty(center, bbox, size).item(), 0)
        right = torch.tensor([.4, .2], requires_grad=True)
        loss = box_penalty(right, bbox, size)
        loss.backward()
        self.assertGreater(right.grad[0].item(), 0)  # Gradient descent moves left.
        self.assertEqual(right.grad[1].item(), 0)
        farther = box_penalty(torch.tensor([.6, .2]), bbox, size)
        self.assertGreater(farther.item(), loss.item())
        self.assertAlmostEqual(loss.item(), box_penalty(torch.tensor([.4, .2]),
            [200,100,600,300], [2000,1000]).item())
        tiny = box_penalty(torch.tensor([.5,.5], requires_grad=True), [1,1,2,2], size)
        self.assertTrue(torch.isfinite(tiny))
        with self.assertRaises(ValueError):
            box_penalty(center, [0,0,0,10], size)

    def test_combined_loss_alignment_and_finite_gradient(self):
        digits = torch.arange(10)
        # Three different coordinate digit lengths, with non-digit separators/EOS.
        for point, sequence in [([5, 50], [5,10,5,0,10]),
                                ([500, 500], [5,0,0,10,5,0,0,10]),
                                ([1000, 0], [1,0,0,0,10,0,10])]:
            labels = torch.tensor([sequence])
            logits = torch.randn(1, len(sequence), 11, requires_grad=True)
            loss, ce, auxiliary, box = coordinate_loss(logits, labels, digits, point,
                auxiliary_weight=1., bbox_pixels=[400,200,600,300], image_size=[1000,500], bbox_weight=.5)
            self.assertTrue(torch.allclose(loss, ce + auxiliary + .5 * box))
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_cleaning_replay_and_selection(self):
        target = lambda screen, point, texts: dict(source_id=screen, target=point, intents=texts,
            bbox_pixels=[0,0,10,10], image_size=[100,100])
        targets = [target('a',[1,1],['Tap button','unique a']),
                   target('a',[2,2],[' TAP  BUTTON ','unique b']),
                   target('b',[2,2],['Tap button'])]
        clean, report = remove_ambiguous_intents(targets)
        self.assertEqual(report['removed_expressions'], 2)
        self.assertEqual(clean[2]['intents'], ['Tap button'])
        self.assertEqual(len(mining_indices(clean, 1)), 2)
        sampler = ReplaySampler(clean, [0], 1)
        for i in range(8):
            example = sampler.next(hard=i % 4 < 2)
            if i % 4 < 2:
                self.assertEqual(example['source_id'], 'a')
                self.assertEqual(example['target'], [1,1])
        self.assertEqual(sampler.counts, {'hard':4,'general':4})
        empty = ReplaySampler(clean, [], 1)
        empty.next(hard=True)
        self.assertEqual(empty.counts['general'], 1)
        self.assertLess(selection_key({'bbox_hits':24,'selection_error':120}, 'bbox'),
                        selection_key({'bbox_hits':23,'selection_error':90}, 'bbox'))
        self.assertLess(selection_key({'bbox_hits':24,'selection_error':90}, 'bbox'),
                        selection_key({'bbox_hits':24,'selection_error':120}, 'bbox'))

    def test_real_split_and_mining_are_training_only(self):
        previous = ROOT / 'runs/qwen2.5-vl-3b-guisyn-v3'
        with tempfile.TemporaryDirectory() as directory:
            targets, validation, split = prepare_design(previous, Path(directory), 6198,
                ROOT / 'data/guisyn-additional/source_annotations.json')
            self.assertEqual(validation, json.loads((previous/'validation.json').read_text()))
            self.assertEqual(split, json.loads((previous/'screenshot_split.json').read_text()))
            clean, report = remove_ambiguous_intents(targets)
            indices = mining_indices(clean, 6215)
            forbidden = set(split['validation_ids'] + split['test_ids'])
            self.assertFalse({t['source_id'] for t in clean} & forbidden)
            self.assertEqual({clean[i]['source_id'] for i in indices}, set(split['train_ids']))
            self.assertEqual(report['removed_expressions'], 231)
            self.assertEqual(json.loads((Path(directory)/'heldout.json').read_text()),
                             json.loads((previous/'heldout.json').read_text()))
            print('Cleaned targets:',len(clean),'intents:',sum(len(t['intents']) for t in clean),
                  'mining candidates:',len(indices),'cleaning:',report)


if __name__ == '__main__':
    unittest.main()
