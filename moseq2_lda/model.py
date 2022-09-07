

from typing import List, Union
from dataclasses import dataclass
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.base import clone

from moseq2_lda.data import MoseqRepresentations

'''
Type specifying a `LinearDiscriminantAnalysis` instance or a `Pipeline` with a final step of type `LinearDiscriminantAnalysis`
'''
LDAEstimator = Union[Pipeline, LinearDiscriminantAnalysis]

def create_lda_pipeline(n_components: int=2, **kwargs) -> LDAEstimator:
    ''' Create a default LDA pipeline

    Parameters:
    n_components (int): number of components to model in the LDA analysis
    **kwargs: additional arguments to be passed to the constructor for class `LinearDiscriminantAnalysis`
    '''
    return Pipeline([
        #('scalar', StandardScaler()),
        ('passthrough', 'passthrough'),
        ('lda', LinearDiscriminantAnalysis(n_components=n_components, solver='eigen', store_covariance=True, **kwargs))
    ])


def set_estimator_params(estimator: LDAEstimator, **kwargs) -> LDAEstimator:
    ''' Set parameters on an LDAEstimator
        helper to handle the case of Pipeline vs LinearDiscriminantAnalysis
    '''
    if isinstance(estimator, Pipeline):
        estimator[-1].set_params(**kwargs)
    else:
        estimator.set_params(**kwargs)
    return estimator


@dataclass
class CrossValidationResult:
    ''' Dataclass containing the results of a cross_validation call
    '''
    ''' Base Estimator '''
    base_estimator: LDAEstimator

    ''' scoring metric used for result calculations '''
    scoring: str
    ''' Name of the parameter which was evaluated '''
    param_name: str
    ''' Unique values considered for parameter `param_name` '''
    param_range: np.ndarray
    ''' Parameters evaluated (`cardinality == test_scores`) '''
    param_values: np.ndarray
    ''' Test scores for each model evaluated '''
    test_scores: np.ndarray
    ''' Train scores for each model evaluated '''
    train_scores: np.ndarray
    ''' Fit times for each model evaluated '''
    fit_times: np.ndarray
    ''' Score times for each model evaluated '''
    score_times: np.ndarray
    ''' Fit estimators for each model evaluated '''
    estimators: List[LDAEstimator]
    ''' An estimate of "chance probability" of correct classification (calc as 1/n_classes) '''
    chance: float

    @property
    def train_scores_mean(self):
        ''' Get the mean of the scoring metric on the train dataset for each parameter value evaluated
        '''
        return np.mean(self.train_scores, axis=1)
    
    @property
    def train_scores_std(self):
        ''' Get the standard deviation of the scoring metric on the train dataset for each parameter value evaluated
        '''
        return np.std(self.train_scores, axis=1)

    @property
    def test_scores_mean(self):
        ''' Get the mean of the scoring metric on the test dataset for each parameter value evaluated
        '''
        return np.mean(self.test_scores, axis=1)

    @property
    def test_scores_std(self):
        ''' Get the standard deviation of the scoring metric on the test dataset for each parameter value evaluated
        '''
        return np.std(self.test_scores, axis=1)

    @property
    def best(self) -> dict:
        ''' Get a dict describing the best model parameter value
        '''
        return self.model_info(np.nanargmax(self.test_scores_mean))

    @property
    def worst(self) -> dict:
        ''' Get a dict describing the worst model parameter value
        '''
        return self.model_info(np.nanargmin(self.test_scores_mean))

    def model_info(self, index: int) -> dict:
        ''' Get a dict describing the a particular model parameter value
        '''
        return {
            'idx': index,
            'param': self.param_range[index],
            'mean': self.test_scores_mean[index],
            'std': self.test_scores_std[index]
        }


    @property
    def param_min(self):
        ''' Get the minimum parameter value that was evaluated
        '''
        return np.min(self.param_range)

    @property
    def param_max(self):
        ''' Get the maximum parameter value that was evaluated
        '''
        return np.max(self.param_range)

    def save(self, dest: str):
        ''' Save this CrossValidationResult

        Parameters:
        dest (str): destination for the saved result
        '''
        joblib.dump(self, dest)

    @classmethod
    def load(cls, path: str) -> 'CrossValidationResult':
        ''' Load a CrossValidationResult from a file

        Parameters:
        dest (str): destination for the saved result
        '''
        return joblib.load(path)



def run_cross_validation(estimator: LDAEstimator, X: np.ndarray, Y: np.ndarray, param_name: str, param_range: np.ndarray, cv=None, scoring='accuracy', n_jobs: int=-1) -> CrossValidationResult:
    ''' Run cross-validation to determine best model hyperparameter value

    Parameters:
    estimator (LDAEstimator): an LDA estimator to cross validate
    X (np.ndarray): data to use for cross validation
    Y (np.ndarray): data labels to use for cross validation
    param_name (str): name of the parameter to optimize
    param_range (np.ndarray): array of parameter values to try
    cv (): Cross-validation generator or an iterable, optional. Determines the cross-validation splitting strategy
    scoring (str): scoring metric to use for model evaluation
    n_jobs (int): number of parallel workers to use for model training
    '''

    if cv is None:
        cv = model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

    base_estimator = clone(estimator)

    test_scores = []
    train_scores = []
    fit_times = []
    score_times = []
    estimators = []
    param_values = []

    for param_value in param_range:
        estimator = set_estimator_params(estimator, **{param_name: param_value})
        cv_results = cross_validate(estimator, X, Y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=True, return_estimator=True)
        param_values.append(param_value)
        test_scores.append(cv_results['test_score'])
        train_scores.append(cv_results['train_score'])
        fit_times.append(cv_results['fit_time'])
        score_times.append(cv_results['score_time'])
        estimators.append(cv_results['estimator'])

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
        chance=(1 / len(set(Y)))
    )

    best = results.best
    best_val_str = best["param"] if isinstance(best["param"], str) else f'{best["param"]:0.2f}'
    print(f'Best value for parameter "{param_name}" is {best_val_str}, ' \
        + f'achieving a mean {scoring} of ~{best["mean"]:0.1%} ± {best["std"]:0.2%} (stdev) on cross-validated data')

    # return estimator trained on full train dataset with best parameters identified via cross validation
    return results


def train_lda_model(estimator, X, Y, split=None):
    pass
    # estimator.fit(X_train, y_train)
    # test_preditions = estimator.predict(X_test)
    # print('Below are performance metrics for estimator using best parameter trained on the entire training dataset and evaluated on held out test data (not used in cross-validation)')
    # print(classification_report(y_true=y_test, y_pred=test_preditions))




def train_lda_pipeline(data: MoseqRepresentations, representation: str, lda_kwargs: dict=None):
    X = getattr(data, representation)
    Y = data.groups
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)

    if lda_kwargs is None:
        lda_kwargs = {}

    estimator = create_lda_pipeline(n_components=2, **lda_kwargs)

    cv_results = run_cross_validation(estimator=estimator,
                      X=X_train,
                      Y=y_train,
                      param_name='shrinkage',
                      param_range=[*list(np.linspace(0, 1, 11, dtype=float)), 'auto'])

    estimator = set_estimator_params(estimator, **{'shrinkage': cv_results.best["param"]})
    estimator.fit(X_train, y_train)
    test_preditions = estimator.predict(X_test)
    print('Below are performance metrics for estimator using best parameter trained on the entire training dataset and evaluated on held out test data (not used in cross-validation)')
    print(classification_report(y_true=y_test, y_pred=test_preditions))



    return LdaPipelineResult(
        data=data,
        representation=representation,
        cv_result=cv_results,
        estimator=estimator,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )


@dataclass
class LdaPipelineResult:
    data: MoseqRepresentations
    representation: str
    cv_result: CrossValidationResult
    estimator: LDAEstimator
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def lda(self):
        if isinstance(self.estimator, Pipeline):
            return self.estimator[-1]
        else:
            return self.estimator

