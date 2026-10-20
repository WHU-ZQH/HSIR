"""CPU numerical and trainer-wiring tests; no TRL/model downloads required."""
import ast
from collections import defaultdict
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "H-GRPO"))
if torch is not None:
    from indiv import attention_spectral_score, normalize_indiv_rewards


def load_method(name, namespace):
    """Execute the actual trainer method with injected lightweight dependencies."""
    tree = ast.parse((ROOT / "H-GRPO/hsir_grpo_trainer.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GRPOTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "H-GRPO/hsir_grpo_trainer.py"), "exec"), namespace)
    return namespace[name]


@unittest.skipIf(torch is None, "Install torch to run H-GRPO numerical tests")
class HGRPOTests(unittest.TestCase):
    def setUp(self):
        self.h = torch.tensor([[1., 0., 2.], [2., 4., -1.], [0., 1., 3.]])
        self.a = torch.tensor([[[1., 0., 0.], [.2, .8, 0.], [.1, .2, .7]],
                               [[1., 0., 0.], [.7, .3, 0.], [.1, .1, .8]]])
        self.mask = torch.ones(3, dtype=torch.long)

    def test_matches_explicit_paper_covariance_and_head_max(self):
        # Independent explicit J_d reference, including ridge used in release.
        H = self.h.double().T
        J = torch.eye(3, dtype=torch.float64) - torch.ones(3, 3, dtype=torch.float64) / 3
        values = torch.linalg.svdvals(H.T @ J @ H + .001 * torch.eye(3, dtype=torch.float64))
        weights = torch.tensor([1., .8, .8], dtype=torch.float64)
        expected = (weights / weights.sum() * values.log()).sum()
        actual = attention_spectral_score(self.h.double(), self.a.double(), self.mask)
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        self.assertGreater(abs(actual.item() - values.log().mean().item()), .01)

    def test_left_and_right_padding_are_excluded(self):
        h = torch.full((6, 3), float("nan"))
        a = torch.full((2, 6, 6), float("nan"))
        h[2:5] = self.h
        a[:, 2:5, 2:5] = self.a
        actual = attention_spectral_score(h, a, torch.tensor([0, 0, 1, 1, 1, 0]))
        torch.testing.assert_close(actual, attention_spectral_score(self.h, self.a, self.mask))

    def test_half_precision_and_rank_deficiency_are_finite(self):
        score = attention_spectral_score(torch.ones(3, 2).half(), self.a.half(), self.mask)
        self.assertTrue(torch.isfinite(score))
        self.assertAlmostEqual(score.item(), torch.tensor(.001).log().item(), places=5)

    def test_group_normalization_negative_raw_scores_and_constant_group(self):
        raw = torch.tensor([-9., -8., -7., -6., 2., 2., 2., 2.])
        expected = torch.tensor([-1., -1/3, 1/3, 1., 0., 0., 0., 0.])
        torch.testing.assert_close(normalize_indiv_rewards(raw, 4), expected)
        torch.testing.assert_close(normalize_indiv_rewards(raw * 3 + 11, 4), expected)

    def test_invalid_inputs_fail_explicitly(self):
        for raw, size in [(torch.ones(3), 2), (torch.tensor([float("nan"), 0.]), 2)]:
            with self.assertRaises(ValueError):
                normalize_indiv_rewards(raw, size)
        with self.assertRaises(ValueError):
            attention_spectral_score(self.h, self.a, torch.zeros(3))
        with self.assertRaises(ValueError):
            attention_spectral_score(self.h, torch.zeros_like(self.a), self.mask)

    def test_medqa_accuracy_accepts_recipe_answer_and_rejects_wrong_option(self):
        tree = ast.parse((ROOT / "H-GRPO/open_r1/rewards.py").read_text(encoding="utf-8"))
        names = {"extract_xml_answer", "accuracy_reward_medical", "tag_count_reward"}
        definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        env = {"re": __import__("re")}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "<rewards>", "exec"), env)
        completions = [[{"content": "<think>\nReasoning\n</think>\n<answer>\nA\n</answer>"}]]
        self.assertEqual(env["accuracy_reward_medical"](completions, ["A"]), [1.])
        self.assertEqual(env["accuracy_reward_medical"](completions, ["B"]), [0.])
        self.assertEqual(env["tag_count_reward"](completions), [1.])

    def test_scoring_uses_matching_middle_block_and_single_sequence_forwards(self):
        calls = []
        def model(**kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["input_ids"].shape[0], 1)
            self.assertTrue(kwargs["output_attentions"])
            # Four blocks: choose hidden[2] and attention[1], not neighboring blocks.
            return SimpleNamespace(
                hidden_states=[torch.full((1, 3, 3), float("nan"))] * 2
                              + [self.h[None]] + [torch.zeros(1, 3, 3)] * 2,
                attentions=[self.a[None] * 0,
                            self.a[None], self.a[None] * 0, self.a[None] * 0],
            )
        method = load_method("_get_indiv_scores", {"attention_spectral_score": attention_spectral_score})
        scores = method(None, model, torch.ones(2, 3, dtype=torch.long), self.mask.repeat(2, 1))
        self.assertEqual(len(calls), 2)
        for score in scores:
            self.assertAlmostEqual(score, attention_spectral_score(self.h, self.a, self.mask).item())
        with self.assertRaisesRegex(RuntimeError, "eager"):
            method(None, lambda **kw: SimpleNamespace(attentions=None),
                   torch.ones(1, 3, dtype=torch.long), self.mask[None])

    def test_actual_reward_pipeline_gathers_before_normalizing_any_reward_order(self):
        # Extract the actual contiguous trainer reward/advantage block. A fake
        # gather models two ranks, with each four-answer group split across ranks.
        tree = ast.parse((ROOT / "H-GRPO/hsir_grpo_trainer.py").read_text(encoding="utf-8"))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_generate_and_score_completions")
        start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == "rewards_per_func")
        end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                   and ast.unparse(n.targets[0]) == "advantages"
                   and isinstance(n.value, ast.Subscript))
        reward_tree = ast.parse((ROOT / "H-GRPO/open_r1/rewards.py").read_text(encoding="utf-8"))
        reward_def = next(n for n in reward_tree.body if isinstance(n, ast.FunctionDef)
                          and n.name == "indiv_score_reward_medical")
        env = {}
        exec(compile(ast.Module(body=[reward_def], type_ignores=[]), "<reward>", "exec"), env)
        indiv_reward = env["indiv_score_reward_medical"]
        indiv_reward.requires_indiv_scores = True
        for rank in (0, 1):
            # Local rank has two answers. Eq. 9 over local pairs would be wrong.
            raw = [-9., -8.] if rank == 0 else [-7., -6.]
            fake = SimpleNamespace(
                reward_funcs=[indiv_reward, lambda **kw: [1., 1.]],
                reward_processing_classes=[None, None],
                reward_weights=torch.tensor([.2, 1.]), num_generations=4,
                accelerator=SimpleNamespace(process_index=rank),
            )
            def gather(local):
                torch.testing.assert_close(local[:, 0], torch.tensor(raw))
                return torch.tensor([[-9., 1.], [-8., 1.], [-7., 1.], [-6., 1.]])
            env = dict(self=fake, torch=torch, nn=torch.nn, device="cpu",
                       prompts=["p", "p"], completions=["a", "b"], inputs=[{}, {}],
                       indiv_scores=raw, gather=gather, normalize_indiv_rewards=normalize_indiv_rewards)
            exec(compile(ast.Module(body=method.body[start:end+1], type_ignores=[]), "<pipeline>", "exec"), env)
            torch.testing.assert_close(env["rewards"], torch.tensor([.8, 1.-.2/3, 1.+.2/3, 1.2]))
            self.assertEqual(env["advantages"].numel(), 2)
            self.assertEqual(bool((env["advantages"] > 0).all()), rank == 1)

    def test_token_loss_uses_global_token_denominator(self):
        namespace = {"torch": torch, "Optional": __import__("typing").Optional,
                     "gather": lambda count: torch.tensor([2, 4])}
        method = load_method("compute_loss", namespace)
        fake = SimpleNamespace(
            _get_per_token_logps=lambda *args: torch.zeros(1, 2, requires_grad=True),
            beta=0., num_iterations=1, epsilon=.2, epsilon_high=.2,
            accelerator=SimpleNamespace(num_processes=2, gather_for_metrics=lambda x: x),
            control=SimpleNamespace(should_evaluate=False), _metrics={"train": defaultdict(list)},
        )
        inputs = dict(prompt_ids=torch.ones(1, 1), prompt_mask=torch.ones(1, 1),
                      completion_ids=torch.ones(1, 2), completion_mask=torch.ones(1, 2),
                      advantages=torch.tensor([3.]), old_per_token_logps=None)
        loss = method(fake, None, inputs)
        self.assertAlmostEqual(loss.item(), -2.)  # -3 * 2 valid tokens * 2 ranks / 6
        loss.backward()


if __name__ == "__main__":
    unittest.main()
