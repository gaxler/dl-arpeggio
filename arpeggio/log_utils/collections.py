class EMACollection:
    def __init__(self, decay: float) -> None:
        self.val = {}
        self.steps = {}

        self._add_val = lambda cur, prev: (1 - decay) * cur + decay * prev

    def add(self, val, tag: str):
        if tag not in self.val:
            self.steps[tag] = 0
            self.val[tag] = val

        prev = self.val[tag]
        self.val[tag] = self._add_val(val, prev)
        self.steps[tag] += 1

        return self.val[tag]
