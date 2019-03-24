class Prediction:
    def __init__(self, prediction, confidence_level):
        self.prediction = prediction
        self.confidence_level = confidence_level

    def __repr__(self):
        percentage = round(self.confidence_level * 100, 2)
        return self.prediction + " (" + str(percentage) + "%)"

    def __eq__(self, other):
        if isinstance(other, Prediction):
            return self.prediction == other.prediction and self.confidence_level == other.confidence_level
        else:
            return self.prediction == other
