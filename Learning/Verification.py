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
    for commit_description, target in test_corpus:
        predicted = predictor.predict(commit_description)
        if predicted != target:
            print(commit_description)
            print("> Predicted", predicted)
            print("> Actual", target)
            print()


    """
    Example of miss classifications:

    {COLLAT}(Openness): Fix few sonar issues
    > Predicted fix
    > Actual refactor

    {COL_BAU}(Collateral): Replace Calendar by LocalDateTime for start date and end date
    > Predicted fix
    > Actual refactor

    {COLLAT_BAU}(SnapshotGeneration): enhance error message
    > Predicted feat
    > Actual fix

    {COL_BAU}(AgreementInfo): integrate agreementInfo into the static data cache
    > Predicted feat
    > Actual refactor

    {COLLAT}(Agreement config): manage errors in the agreement cache
    > Predicted fix
    > Actual feat

    {COL_BAU}(collat_algo_service target refactoring) Refactor to be able to have a mock dll to be used for other target unit tests.
    > Predicted refactor
    > Actual feat

    {COL_BAU}(Openness cleaning): Clean deprecated code after openess development. Remove all backward compatibility code (except migration code).
    > Predicted refactor
    > Actual fix

    {COL_BAU}(Openness cleaning): Clean deprecated code after openess development. Remove valuationContext classes.
    > Predicted refactor
    > Actual fix
    """

    # What the fuck fixes

    """
    [COLLATERAL](Fix): reference returned instead of copy
    """
