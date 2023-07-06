"""Module contains functions for modelling moseq data with LDA."""
from typing import List, Optional, Sequence, Union
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.base import clone

from moseq2_lda.data import MoseqRepresentations, RepresentationType

"""
Type specifying a `LinearDiscriminantAnalysis` instance or a `Pipeline` with a final step of type `LinearDiscriminantAnalysis`
"""
LDAEstimator = Union[Pipeline, LinearDiscriminantAnalysis]


def create_lda_pipeline(**kwargs) -> LDAEstimator:
    """Create a default LDA pipeline.

    Args:
        **kwargs: additional arguments to be passed to the constructor for class `LinearDiscriminantAnalysis`
    """
    n_components = kwargs.pop("n_components", 2)
    return Pipeline(
        [
            # ('scalar', StandardScaler()),
            # ("passthrough", "passthrough"),
            ("lda", LinearDiscriminantAnalysis(n_components=n_components, solver="eigen", store_covariance=True, **kwargs)),
        ]
    )


def set_estimator_params(estimator: LDAEstimator, **kwargs) -> LDAEstimator:
    """Set parameters on an LDAEstimator.

    Helper to handle the case of Pipeline vs LinearDiscriminantAnalysis

    Args:
        estimator (LDAEstimator): estimator to operate upon
        kwargs: additional parameters to set on the LDA estimator
    """
    if isinstance(estimator, Pipeline):
        estimator[-1].set_params(**kwargs)
    else:
        estimator.set_params(**kwargs)
    return estimator


@dataclass
class CrossValidationResult:
    """Dataclass containing the results of a cross_validation call."""

    """ Base Estimator """
    base_estimator: LDAEstimator

    """ Scoring metric used for result calculations """
    scoring: str
    """ Name of the parameter which was evaluated """
    param_name: str
    """ Unique values considered for parameter `param_name` """
    param_range: Sequence[Union[str, int, float]]
    """ Parameters evaluated (`cardinality == test_scores`) """
    param_values: np.ndarray
    """ Test scores for each model evaluated """
    test_scores: np.ndarray
    """ Train scores for each model evaluated """
    train_scores: np.ndarray
    """ Fit times for each model evaluated """
    fit_times: np.ndarray
    """ Score times for each model evaluated """
    score_times: np.ndarray
    """ Fit estimators for each model evaluated """
    estimators: List[LDAEstimator]
    """ An estimate of "chance probability" of correct classification (calc as 1/n_classes) """
    chance: float

    @property
    def train_scores_mean(self):
        """Get the mean of the scoring metric on the train dataset for each parameter value evaluated."""
        return np.mean(self.train_scores, axis=1)

    @property
    def train_scores_std(self):
        """Get the standard deviation of the scoring metric on the train dataset for each parameter value evaluated."""
        return np.std(self.train_scores, axis=1)

    @property
    def test_scores_mean(self):
        """Get the mean of the scoring metric on the test dataset for each parameter value evaluated."""
        return np.mean(self.test_scores, axis=1)

    @property
    def test_scores_std(self):
        """Get the standard deviation of the scoring metric on the test dataset for each parameter value evaluated."""
        return np.std(self.test_scores, axis=1)

    @property
    def best(self) -> dict:
        """Get a dict describing the best model parameter value."""
        return self.model_info(int(np.nanargmax(self.test_scores_mean)))

    @property
    def worst(self) -> dict:
        """Get a dict describing the worst model parameter value."""
        return self.model_info(int(np.nanargmin(self.test_scores_mean)))

    def model_info(self, index: int) -> dict:
        """Get a dict describing the a particular model parameter value."""
        return {"idx": index, "param": self.param_range[index], "mean": self.test_scores_mean[index], "std": self.test_scores_std[index]}

    @property
    def param_min(self):
        """Get the minimum parameter value that was evaluated."""
        return np.min(self.param_range)

    @property
    def param_max(self):
        """Get the maximum parameter value that was evaluated."""
        return np.max(self.param_range)

    def save(self, dest: str):
        """Save this CrossValidationResult.

        Args:
            dest (str): destination for the saved result
        """
        joblib.dump(self, dest)

    @classmethod
    def load(cls, path: str) -> "CrossValidationResult":
        """Load a CrossValidationResult from a file.

        Args:
            path (str): path to the pickle file to load
        """
        return joblib.load(path)


def run_cross_validation(
    estimator: LDAEstimator,
    X: np.ndarray,
    Y: np.ndarray,
    param_name: str,
    param_range: Sequence[Union[str, int, float]],
    cv=None,
    scoring="accuracy",
    n_jobs: int = -1,
) -> CrossValidationResult:
    """Run cross-validation to determine best model hyperparameter value.

    Args:
        estimator (LDAEstimator): an LDA estimator to cross validate
        X (np.ndarray): data to use for cross validation
        Y (np.ndarray): data labels to use for cross validation
        param_name (str): name of the parameter to optimize
        param_range (np.ndarray): array of parameter values to try
        cv (): Cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy
        scoring (str): scoring metric to use for model evaluation
        n_jobs (int): number of parallel workers to use for model training
    """
    if cv is None:
        cv = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    base_estimator = clone(estimator)

    test_scores = []
    train_scores = []
    fit_times = []
    score_times = []
    estimators = []
    param_values = []

    cv_kwargs = {"cv": cv, "scoring": scoring, "n_jobs": n_jobs, "return_train_score": True, "return_estimator": True}

    for param_value in param_range:
        estimator = set_estimator_params(estimator, **{param_name: param_value})
        cv_results = model_selection.cross_validate(estimator, X, Y, **cv_kwargs)
        param_values.append(param_value)
        test_scores.append(cv_results["test_score"])
        train_scores.append(cv_results["train_score"])
        fit_times.append(cv_results["fit_time"])
        score_times.append(cv_results["score_time"])
        estimators.append(cv_results["estimator"])

    results = CrossValidationResult(
        base_estimator=base_estimator,
        scoring=scoring,
        param_name=param_name,
        param_range=param_range,
        param_values=np.array(param_values),
        test_scores=np.array(test_scores),
        train_scores=np.array(train_scores),
        fit_times=np.array(fit_times),
        score_times=np.array(score_times),
        estimators=estimators,
        chance=(1 / len(set(Y))),
    )

    best = results.best
    best_val_str = best["param"] if isinstance(best["param"], str) else f'{best["param"]:0.2f}'
    print(f'Best value for parameter "{param_name}" is {best_val_str}')
    print(f'Achieving a mean {scoring} of ~{best["mean"]:0.1%} ± {best["std"]:0.2%} (stdev) on cross-validated data')

    return results


def train_lda_model(estimator: LDAEstimator, data: MoseqRepresentations, representation: RepresentationType) -> "LdaResult":
    """Train a LDA model on a representation from a dataset.

    Args:
        estimator (LDAEstimator): estimator to train
        data (MoseqRepresentations): dataset to use for training
        representation (RepresentationType): type of representation to use for training
    """
    estimator.fit(data.data(representation), data.groups)

    return LdaResult(estimator=estimator, data=data, representation=representation)


@dataclass
class LdaResult:
    """Encapuslates the results of training a single LDA model."""

    estimator: LDAEstimator
    data: MoseqRepresentations
    representation: RepresentationType

    @property
    def lda(self) -> LinearDiscriminantAnalysis:
        """Get the underlying instance of `LinearDiscriminantAnalysis`."""
        if isinstance(self.estimator, Pipeline):
            return self.estimator[-1]
        else:
            return self.estimator

    def _get_data(self, data: Optional[MoseqRepresentations] = None):
        if data is not None:
            x = data.data(self.representation)
            y = data.groups
        else:
            x = self.data.data(self.representation)
            y = self.data.groups

        return x, y

    def predict(self, data: Optional[MoseqRepresentations] = None):
        """Predict class from `data`.

        Args:
            data (MoseqRepresentations|None): Data to predict upon. If None, use `self.data`
        """
        x, _ = self._get_data(data)
        return self.lda.predict(x)

    def transform(self, data: Optional[MoseqRepresentations] = None):
        """Transform `data`.

        Args:
            data (MoseqRepresentations|None): Data to transform. If None, use `self.data`
        """
        x, _ = self._get_data(data)
        return self.lda.transform(x)

    def score(self, data: Optional[MoseqRepresentations] = None):
        """Get the LDA score for `data`.

        Args:
            data (MoseqRepresentations|None): Data to score. If None, use `self.data`
        """
        x, y = self._get_data(data)
        return self.lda.score(x, y)

    def classification_report(self, data: Optional[MoseqRepresentations] = None):
        """Generate a classification report for `data`.

        Args:
            data (MoseqRepresentations|None): Data for which to generate a classification report. If None, use `self.data`.
        """
        x, y = self._get_data(data)
        p = self.lda.predict(x)
        return classification_report(y_true=y, y_pred=p)

    def save(self, dest: str):
        """Save this LdaResult.

        Args:
            dest (str): destination for the saved result
        """
        joblib.dump(self, dest)

    @classmethod
    def load(cls, path: str) -> "LdaResult":
        """Load a LdaResult from a file.

        Args:
            path (str): path to the pickle file to load
        """
        return joblib.load(path)


def train_lda_pipeline(
    data: MoseqRepresentations, representation: RepresentationType, holdout: float = 0.3, lda_kwargs: Optional[dict] = None
) -> "LdaPipelineResult":
    """This is a "batteries-included" method to train an LDA model.

    Performs the following procedure:

    - split the representations into `test` and `train` subsets
    - creates an LDA estimator
    - run a cross-validated search (k-fold stratified CV) for the hyperparameter `shrinkage` using \
      only the `train` subset of the representations
    - select the best hyperparameter value from the search, and then train the classifier using \
      the full `train` subset of the representations
    - predict on the held-out `test` subsets and print a classification report
    - construct and return a `LdaPipelineResult` object

    Args:
        data (MoseqRepresentations): dataset to use for training
        representation (RepresentationType): Representation to use
        holdout (float): percentage of data to hold out during training
        lda_kwargs (dict): additional kwargs to pass to `create_lda_pipeline()`
    """
    # Split data into train and test sets.
    # Train will be used for CV and final model training
    # Test will only be used for evaluation of the final model
    train, test = data.split(holdout)

    if lda_kwargs is None:
        lda_kwargs = {}

    # Create a LDA pipeline, passing along any kwargs the user supplied
    estimator = create_lda_pipeline(**lda_kwargs)

    # Run cross validation using the estimator
    cv_results = run_cross_validation(
        estimator=estimator,
        X=train.data(representation),
        Y=train.groups,
        param_name="shrinkage",
        param_range=[*list(np.linspace(0, 1, 11, dtype=float)), "auto"],
    )

    estimator = set_estimator_params(estimator, **{"shrinkage": cv_results.best["param"]})
    final = train_lda_model(estimator, train, representation)
    print(
        "Below are performance metrics for estimator using best parameter trained on the entire training dataset "
        "and evaluated on held out test data (not used in cross-validation)"
    )
    print(final.classification_report(test))

    return LdaPipelineResult(data=data, representation=representation, cv_result=cv_results, final=final, train=train, test=test)


@dataclass
class LdaPipelineResult:
    """Class containing results from `train_lda_pipeline`."""

    data: MoseqRepresentations
    representation: str
    cv_result: CrossValidationResult
    final: LdaResult
    train: MoseqRepresentations
    test: MoseqRepresentations

    def save(self, dest: str):
        """Save this LdaPipelineResult.

        Args:
            dest (str): destination for the saved result
        """
        joblib.dump(self, dest)

    @classmethod
    def load(cls, path: str) -> "LdaPipelineResult":
        """Load a LdaPipelineResult from a file.

        Args:
            path (str): path to the pickle file to load
        """
        return joblib.load(path)
