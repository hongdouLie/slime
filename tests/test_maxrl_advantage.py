import unittest

import torch

from custom_advantages.maxrl import compute_maxrl_sequence_advantages


class MaxRLAdvantageTest(unittest.TestCase):
    def assert_tensor_close(self, actual, expected):
        self.assertTrue(
            torch.allclose(actual, expected, atol=2e-5),
            msg=f"\nactual:   {actual}\nexpected: {expected}",
        )

    def test_one_success_out_of_four(self):
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
        actual = compute_maxrl_sequence_advantages(
            rewards,
            group_size=4,
        )
        expected = torch.tensor([3.0, -1.0, -1.0, -1.0])
        self.assert_tensor_close(actual, expected)

    def test_two_successes_out_of_four(self):
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        actual = compute_maxrl_sequence_advantages(
            rewards,
            group_size=4,
        )
        expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
        self.assert_tensor_close(actual, expected)

    def test_one_success_out_of_eight(self):
        rewards = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        actual = compute_maxrl_sequence_advantages(
            rewards,
            group_size=8,
        )
        expected = torch.tensor(
            [7.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        )
        self.assert_tensor_close(actual, expected)

    def test_all_wrong_group_is_zero(self):
        actual = compute_maxrl_sequence_advantages(
            torch.zeros(4),
            group_size=4,
        )
        self.assertTrue(torch.equal(actual, torch.zeros(4)))

    def test_all_correct_group_is_zero(self):
        actual = compute_maxrl_sequence_advantages(
            torch.ones(4),
            group_size=4,
        )
        self.assert_tensor_close(actual, torch.zeros(4))

    def test_multiple_groups_remain_separate(self):
        rewards = torch.tensor([
            1.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 1.0, 0.0,
        ])
        actual = compute_maxrl_sequence_advantages(
            rewards,
            group_size=4,
        )
        expected = torch.tensor([
            3.0, -1.0, -1.0, -1.0,
            1.0, -1.0, 1.0, -1.0,
        ])
        self.assert_tensor_close(actual, expected)

    def test_invalid_reward_count(self):
        with self.assertRaises(ValueError):
            compute_maxrl_sequence_advantages(
                torch.tensor([1.0, 0.0, 0.0]),
                group_size=4,
            )


if __name__ == "__main__":
    unittest.main()
