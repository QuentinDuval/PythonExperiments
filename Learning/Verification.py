from Learning.Predictor import *


def verify_predictor(predictor: Predictor):
    tests = {
        "fix":
            [
                "quantity was wrong",
                "empty capital structures are not saved anymore",
                "introduce crash on purpose"        # The joke
            ],
        "feat":
            [
                "add new screen for collateral agreements",
                "improve performance of cash sweeping"
            ],
        "refactor":
            [
                "refactor screen of collateral agreements",
                "move CollateralAgreement class to new folder",
                "extract computeQuantity from Trade class",
                "use smart pointers to simplify memory management",
                "clean code in computeQuantity",
                "remove deprecated code in TradeUtilities"
            ]
    }

    failed = []
    for target, examples in tests.items():
        for example in examples:
            predicted = predictor.predict(example)
            if not target == predicted:
                failed.append({"commit": example, "predicted": predicted, "expected": target})
    return failed


def show_errors(predictor: Predictor, test_corpus: CommitMessageCorpus):
    for commit in test_corpus.classified:
        predicted = predictor.predict(commit.message)
        if predicted != commit.classification:
            print(commit.raw_message)
            print("> Predicted", predicted)
            print("> Actual", commit.classification)
            print()
