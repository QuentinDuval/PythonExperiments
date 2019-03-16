class Ratio:
    def __init__(self, correct=0, total=0):
        self.correct = correct
        self.total = total

    def __add__(self, other):
        return Ratio(self.correct + other.correct, self.total + other.total)

    def __str__(self):
        return str(self.correct) + "/" + str(self.total) + " (" + str(self.to_percentage() * 100) + "%)"

    def to_percentage(self):
        return self.correct / self.total if self.total else 0

    def __lt__(self, other):
        return self.to_percentage() < other.to_percentage()

    def __gt__(self, other):
        return self.to_percentage() > other.to_percentage()
