from scipy.stats import norm
import unittest

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections

    def adjusted_rate(self, count, total):
        """Prevents hit or false alarm rates from being exactly 0 or 1."""
        return (count + 0.5) / (total + 1)

    def d_prime(self):
        hit_rate = self.adjusted_rate(self.hits, self.hits + self.misses)
        fa_rate = self.adjusted_rate(self.falseAlarms, self.falseAlarms + self.correctRejections)
        return norm.ppf(hit_rate) - norm.ppf(fa_rate)

    def criterion(self):
        hit_rate = self.adjusted_rate(self.hits, self.hits + self.misses)
        fa_rate = self.adjusted_rate(self.falseAlarms, self.falseAlarms + self.correctRejections)
        return -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))

# Unit Testing
class TestSignalDetection(unittest.TestCase):
    def test_sdt(self):
        sdt = SignalDetection(50, 50, 30, 70)
        print(f"Testing d' and C with test case (50,50,30,70)")
        print(f"Computed d': {sdt.d_prime()}")
        print(f"Computed criterion: {sdt.criterion()}")
        self.assertAlmostEqual(sdt.d_prime(), 1.0, places=2)

if __name__ == "__main__":
    unittest.main()

