"""Replay row-exposure credit earned by new unique training states.

The ratio is a sampling budget, not a claim that a particular number of passes
is optimal. Decimal accounting preserves fractional credit across checkpoints.
Legacy checkpoints start without retroactive credit: their exact historical
row exposure cannot be recovered from optimizer steps or a retained buffer.
"""
from decimal import Decimal, InvalidOperation, localcontext
import math


class ReplayExposure:
    VERSION = 1

    def __init__(self, ratio, *, legacy_states=0):
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or ratio < 0:
            raise ValueError("Replay ratio must be finite and nonnegative")
        if type(legacy_states) is not int or legacy_states < 0:
            raise ValueError("Legacy training-state count must be nonnegative")
        self.ratio = Decimal(str(ratio))
        self.credit = Decimal(0)
        self.unique_states = 0
        self.trained_examples = 0
        self.legacy_states = legacy_states
        self.migration = ("Legacy checkpoint: retained replay and known training states receive zero retroactive credit; "
                          "historical row exposure and pending untrained rows are not reliably reconstructible."
                          if legacy_states else None)

    @property
    def available(self):
        return int(self.credit)

    @staticmethod
    def _balance(ratio, unique, trained):
        # Avoid the process-global Decimal precision silently rounding away a
        # debit for a large ratio, or accumulated fractional credit for a tiny
        # one. Every finite CLI float is an exact decimal rational here.
        digits, exponent = len(ratio.as_tuple().digits), ratio.as_tuple().exponent
        with localcontext() as context:
            context.prec = max(28, digits + abs(exponent) + len(str(unique)) + len(str(trained)) + 4)
            return ratio * unique - trained

    def admit(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("New unique training-state count must be nonnegative")
        self.unique_states += count
        self.credit = self._balance(self.ratio, self.unique_states, self.trained_examples)

    def debit(self, trained_examples):
        if type(trained_examples) is not int or not 0 <= trained_examples <= self.available:
            raise ValueError("Actual trained examples exceed available replay exposure credit")
        self.trained_examples += trained_examples
        self.credit = self._balance(self.ratio, self.unique_states, self.trained_examples)

    def to_dict(self):
        return {"version": self.VERSION, "ratio": str(self.ratio), "credit": str(self.credit),
                "new_unique_training_states": self.unique_states,
                "trained_examples": self.trained_examples,
                "legacy_uncredited_training_states": self.legacy_states,
                "migration": self.migration}

    @classmethod
    def restore(cls, saved, ratio):
        if not isinstance(saved, dict) or type(saved.get("version")) is not int or saved["version"] != cls.VERSION:
            raise ValueError("Unsupported replay exposure ledger")
        result = cls(ratio, legacy_states=saved.get("legacy_uncredited_training_states", 0))
        try:
            stored_ratio, credit = Decimal(saved["ratio"]), Decimal(saved["credit"])
        except (KeyError, TypeError, ValueError, InvalidOperation) as error:
            raise ValueError("Malformed replay exposure ledger") from error
        if not stored_ratio.is_finite() or stored_ratio != result.ratio:
            raise ValueError("Resume replay ratio must match the saved exposure ledger")
        unique, trained = saved.get("new_unique_training_states"), saved.get("trained_examples")
        if (type(unique) is not int or unique < 0 or type(trained) is not int or trained < 0 or
                not credit.is_finite() or credit < 0 or credit != cls._balance(stored_ratio, unique, trained)):
            raise ValueError("Inconsistent replay exposure ledger accounting")
        if saved.get("migration") is not None and not isinstance(saved["migration"], str):
            raise ValueError("Malformed replay exposure migration note")
        result.credit, result.unique_states, result.trained_examples = credit, unique, trained
        result.migration = saved.get("migration")
        return result
