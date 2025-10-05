import time
import sys

class Progress:
    def __init__(self, total, bar_len=30):
        self.total = int(total)
        self.done = 0
        self.start = time.time()
        self.bar_len = bar_len

    def step(self, n=1):
        self.done += n
        pct = self.done / self.total
        elapsed = time.time() - self.start
        rate = self.done / max(elapsed, 1e-9)
        rem = (self.total - self.done) / max(rate, 1e-9)
        filled = int(self.bar_len * pct)
        bar = "#" * filled + "-" * (self.bar_len - filled)
        sys.stdout.write(f"\r[{bar}] {self.done}/{self.total} ({pct*100:5.1f}%) ETA {rem:6.1f}s")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\n")
