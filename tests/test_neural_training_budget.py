import unittest

from ghqnet.training_budget import ReplayExposure


class ReplayExposureTests(unittest.TestCase):
    def test_fractional_credit_survives_serialization_without_rounding_loss(self):
        ledger = ReplayExposure(.1)
        for _ in range(9):
            ledger.admit(1)
            ledger = ReplayExposure.restore(ledger.to_dict(), .1)
            self.assertEqual(ledger.available, 0)
        ledger.admit(1)
        self.assertEqual(ledger.available, 1)
        ledger.debit(1)
        self.assertEqual(ledger.to_dict()['credit'], '0.0')
        self.assertEqual(ledger.trained_examples, 1)

    def test_credit_depends_on_new_states_and_actual_debits_not_save_count(self):
        first, second = ReplayExposure(1.25), ReplayExposure(1.25)
        first.admit(12)
        for _ in range(12):
            second.admit(1)
            second = ReplayExposure.restore(second.to_dict(), 1.25)
        self.assertEqual(first.to_dict(), second.to_dict())
        first.debit(4)
        first = ReplayExposure.restore(first.to_dict(), 1.25)
        self.assertEqual(first.available, 11)
        self.assertEqual(first.trained_examples, 4)
        first.admit(0)
        self.assertEqual(first.available, 11)

    def test_legacy_states_receive_no_retroactive_credit(self):
        ledger = ReplayExposure(1, legacy_states=86000)
        self.assertEqual(ledger.available, 0)
        self.assertIn('zero retroactive credit', ledger.migration)
        ledger.admit(2)
        self.assertEqual(ledger.available, 2)

    def test_large_credit_keeps_single_row_debits_exact(self):
        ledger = ReplayExposure(1e30)
        ledger.admit(1)
        ledger.debit(1)
        self.assertEqual(ledger.available, 10**30-1)
        self.assertEqual(ReplayExposure.restore(ledger.to_dict(), 1e30).available, 10**30-1)

    def test_invalid_ratios_debits_or_saved_accounting_are_rejected(self):
        for ratio in (-1, float('nan'), float('inf'), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                ReplayExposure(ratio)
        ledger = ReplayExposure(.5)
        ledger.admit(3)
        for debit in (-1, 2, True, 1.5):
            with self.subTest(debit=debit), self.assertRaises(ValueError):
                ledger.debit(debit)
        with self.assertRaisesRegex(ValueError, 'ratio must match'):
            ReplayExposure.restore(ledger.to_dict(), 1)
        for change in ({'credit': '-1'}, {'credit': 'NaN'}, {'credit': '2'},
                       {'trained_examples': True}, {'new_unique_training_states': -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ReplayExposure.restore({**ledger.to_dict(), **change}, .5)


if __name__ == '__main__':
    unittest.main()
