class EloSystem:
    def __init__(self, base=1000, k=32):
        self.base = base
        self.k = k
        self.ratings = {}

    def get(self, id):
        return self.ratings.get(id, self.base)

    def expected(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, a, b, result):
        ra, rb = self.get(a), self.get(b)

        ea = self.expected(ra, rb)

        self.ratings[a] = ra + self.k * (result - ea)
        self.ratings[b] = rb + self.k * ((1 - result) - (1 - ea))