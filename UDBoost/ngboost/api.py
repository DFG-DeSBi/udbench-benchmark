"The NGBoost library API"
# pylint: disable=too-many-arguments

# - SKlearn - #
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import resample
from sklearn.utils import check_random_state

# - Numpy - #
import numpy as np

# - NGBoost - #
from ngboost.distns import (
    Bernoulli,
    ClassificationDistn,
    LogNormal,
    Normal,
    RegressionDistn,
)
from ngboost.distns.utils import SurvivalDistnClass
from ngboost.helpers import Y_from_censored
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from .ngboost_core import NGBoost
from ngboost.scores import LogScore
from .distns.utils import epistemic_scaling


class NGBRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost regression models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values.

    Parameters:
        Dist : ngboost.distns
            Assumed distributional form of Y|X=x, e.g. `Normal`.

        Score : ngboost.scores
            Rule to compare probabilistic predictions Ŷ to the observed data y, e.g. `LogScore`.

        base_criterion : str, default="friedman_mse"
            Loss used to evaluate splits in the base learner.
            Options include `"squared_error"`, `"friedman_mse"`, `"absolute_error"`, `"poisson"`.

        splitter : {"best", "random"}, default="best"
            Strategy used to choose the split at each node.

        alpha : float, default=0.0
            Complexity-pruning parameter (ccp_alpha) for the base trees.

        min_samples_leaf : int or float, default=1
            Minimum number (or fraction) of samples required to be at a leaf node.

        min_samples_split : int or float, default=2
            Minimum number (or fraction) of samples required to split an internal node.

        max_depth : int, default=6
            Maximum depth of the individual regression estimators.

        min_weight_fraction_leaf : float, default=0.0
            Minimum weighted fraction of the sum total of weights required to be at a leaf node.

        max_features : int, float, {"sqrt", "log2"} or None, default=None
            Number of features to consider when looking for the best split.

        max_leaf_nodes : int or None, default=None
            Grow trees with `max_leaf_nodes` in best-first fashion if not None.

        min_impurity_decrease : float, default=0.0
            A node will be split if this split induces a decrease of the impurity
            greater than or equal to this value.

        monotone_cst : array-like of shape (n_features,) or None, default=None
            Monotonicity constraints per feature: 1 for increasing,
            -1 for decreasing, 0 for no constraint.

        natural_gradient : bool, default=True
            Whether to use the natural gradient in the boosting updates.

        n_estimators : int, default=500
            The number of boosting iterations to fit.

        learning_rate : float, default=0.01
            Learning rate that shrinks the contribution of each learner.

        minibatch_frac : float, default=1.0
            Fraction of samples to subsample for each boosting iteration.

        col_sample : float, default=1.0
            Fraction of features to subsample for each boosting iteration.

        verbose : bool, default=True
            Whether to print progress messages during fitting.

        verbose_eval : int, default=100
            Print evaluation metrics at every `verbose_eval` iterations.

        tol : float, default=1e-4
            Numerical tolerance for early stopping criterion.

        random_state : int, RandomState instance or None, default=None
            Controls the random seed for reproducibility.

        validation_fraction : float, default=0.1
            Fraction of training data to set aside as validation set for early stopping.

        early_stopping_rounds : int or None, default=None
            Number of consecutive iterations with no improvement to trigger early stopping.
            Set to None to disable early stopping.

        SGLB : bool, default=False
            Whether to use Stochastic Gradient Langevin Boosting (SGLB).

        langevin_noise_scale : float, default=1
            Scale of the Langevin noise added during training (only if `SGLB=True`).

        n_regressors : int, default=1
            The number of base regressors to use in the ensemble.

        metadistribution_method : str or None, default=None
            The method to use for the metadistribution.
            Options include:
                - "ensemble_SGLB"
                - "virtual_SGLB"
                - "SGB"
                - "evidential_regression"
                - "bagging"
                - "KGB"
                - None (no metadistribution)

        bagging_frac : float, default=1.0
            The fraction of samples to use for bagging (subsampling) in the ensemble.

        epistemic_scaling : bool, default=False
            Whether to apply additional scaling to the epistemic (model) uncertainty estimates.

    Output:
        An NGBRegressor object or an NGBoost ensemble that can be fit using .fit(X, y)
        and used to predict distributions, means, variances, etc.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        base_criterion="friedman_mse",
        splitter='best',
        alpha=0.0,
        min_samples_leaf=1,
        min_samples_split=2,
        max_depth=6,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        monotone_cst=None,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        bagging_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        SGLB=False,
        langevin_noise_scale=1,
        n_regressors=1,
        metadistribution_method=None,
        epistemic_scaling=None,
        replace=False,
    ):
        
        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."

        if not hasattr(
            Dist, "scores"
        ):  # user is trying to use a dist that only has censored scores implemented
            Dist = Dist.uncensor(Score)

        self.Dist = Dist
        self.Score = Score
        self.base_criterion = base_criterion
        self.splitter = splitter
        self.alpha = alpha
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.monotone_cst = monotone_cst
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.bagging_frac = bagging_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.tol = tol
        self.random_state = random_state
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.SGLB = SGLB
        self.langevin_noise_scale = langevin_noise_scale
        self.metadistribution_method = metadistribution_method
        self.epistemic_scaling = epistemic_scaling
        self.replace = replace
        # ------------- #

        # --- API Attributes --- #
        self._estimator_type = "regressor"
        self.ensemble_models = []
        self.n_regressors = n_regressors


    def __getstate__(self):
        state = super().__getstate__()

        # Handle DistWithUncensoredScore safely
        if hasattr(self, "Dist") and getattr(self.Dist, "__name__", "") == "DistWithUncensoredScore":
            # Replace with base class for pickling
            state["Dist"] = getattr(self.Dist, "__base__", self.Dist)
            state["uncensor"] = True

        # Optionally remove unpicklable fields safely
        for key in ["Manifold", "some_other_unpicklable_field"]:
            state.pop(key, None)  # Remove if present, skip if not

        return state


    def __setstate__(self, state_dict):
        if "uncensor" in state_dict:
            if hasattr(state_dict["Dist"], "uncensor"):
                state_dict["Dist"] = state_dict["Dist"].uncensor(state_dict["Score"])
            del state_dict["uncensor"]  # Clean up after use
        super().__setstate__(state_dict)



    def pred_uncertainty(self, X, mode: str = 'bayesian_kl'):
        """
        Computes predictive, epistemic, and aleatoric uncertainties for NGBoost ensembles.
        Supports several uncertainty estimation strategies, including distribution-based logic.

        Parameters:
            X (array-like): Input data for which to compute uncertainties
            mode (str): Method used to compute uncertainty. Options:
                - 'distribution' (uses distribution object logic)
                - 'bayesian_mean'
                - 'bayesian_kl'
                - 'levi_simple'
                - 'levi_kl'

        Returns:
            dict: Dictionary containing uncertainty estimates:
                - mean: Mean prediction across all models or distribution
                - aleatoric: Aleatoric uncertainty
                - epistemic: Epistemic uncertainty
                - predictive: Total uncertainty (aleatoric + epistemic)
                - aleatoric_lower_bound: Lower aleatoric bound (if applicable)
                - aleatoric_upper_bound: Upper aleatoric bound (if applicable)
                - predictive_lower_bound: Lower total uncertainty (if applicable)
                - predictive_upper_bound: Upper total uncertainty (if applicable)
        """
        if self.metadistribution_method == 'evidential_regression':
            dist = self.pred_dist(X)
            if hasattr(self.Dist, "pred_uncertainty") and getattr(dist, "is_EDL", False):
                if getattr(dist, "epistemic_scaling", None):
                    return epistemic_scaling(dist, knn=10)
                return dist.pred_uncertainty()
            else:
                raise NotImplementedError(
                    "The distribution does not implement a compatible pred_uncertainty method."
                )

        # --- Ensemble-based Uncertainty Estimation --- #
        if self.metadistribution_method == "virtual_SGBL":
            staged_predictions = self.ensemble_models[0].staged_pred_dist(X)
            indices = np.linspace(
                len(staged_predictions) // 2,
                len(staged_predictions) - 1,
                self.n_regressors,
                dtype=int
            )
            parameters = np.array([staged_predictions[i].params for i in indices])
        else:
            parameters = np.array([model.pred_dist(X).params for model in self.ensemble_models])

        locs = np.array([param['loc'] for param in parameters])
        scales = np.array([param['scale'] for param in parameters])
        
        mean_prediction = np.mean(locs, axis=0)

        # Initialize optional values
        aleatoric_uncertainty = None
        epistemic_uncertainty = None
        aleatoric_uncertainty_lower_bound = None
        aleatoric_uncertainty_upper_bound = None

        match mode:
            case 'bayesian_mean':
                aleatoric_uncertainty = np.mean(scales, axis=0)
                epistemic_uncertainty = np.var(locs, axis=0)

            case 'levi_simple':
                aleatoric_uncertainty_lower_bound = np.min(scales, axis=0)
                aleatoric_uncertainty_upper_bound = np.max(scales, axis=0)

                pairwise_diff = locs[:, None, :] - locs[None, :, :]
                epistemic_uncertainty = np.max(np.abs(pairwise_diff), axis=(0, 1))

            case 'bayesian_kl':
                log_scales_squared = np.log(scales ** 2)
                aleatoric_uncertainty = 0.5 * (1 + np.log(2 * np.pi) + np.mean(log_scales_squared, axis=0))

                mean_mu = np.mean(locs, axis=0)
                mean_sigma = np.mean(scales, axis=0)
                log_mean_sigma = np.log(mean_sigma ** 2)

                mean_mu_deviation = np.mean((locs ** 2 - mean_mu ** 2), axis=0) / (mean_sigma ** 2)
                epistemic_uncertainty = 0.5 * (log_mean_sigma - np.mean(log_scales_squared, axis=0) + mean_mu_deviation)

            case 'levi_kl':
                entropies = 0.5 * np.log(2 * np.pi * np.e * scales ** 2)
                aleatoric_uncertainty_lower_bound = np.min(entropies, axis=0)
                aleatoric_uncertainty_upper_bound = np.max(entropies, axis=0)

                divergences = []
                for p in parameters:
                    for q in parameters:
                        kl = (
                            np.log(q['scale']) - np.log(p['scale']) +
                            (p['scale'] ** 2 + (p['loc'] - q['loc']) ** 2) / (2 * q['scale'] ** 2) - 0.5
                        )
                        divergences.append(kl)
                divergences = np.array(divergences)
                epistemic_uncertainty = np.max(divergences, axis=0)

            case _:
                raise ValueError(f"Unknown uncertainty mode: '{mode}'")

        return {
            'mean': mean_prediction,
            'predictive': (
                aleatoric_uncertainty + epistemic_uncertainty
                if aleatoric_uncertainty is not None else None
            ),
            'predictive_upper_bound': (
                aleatoric_uncertainty_upper_bound + epistemic_uncertainty
                if aleatoric_uncertainty_upper_bound is not None else None
            ),
            'predictive_lower_bound': (
                aleatoric_uncertainty_lower_bound + epistemic_uncertainty
                if aleatoric_uncertainty_lower_bound is not None else None
            ),
            'aleatoric': aleatoric_uncertainty,
            'epistemic': epistemic_uncertainty,
            'aleatoric_lower_bound': aleatoric_uncertainty_lower_bound,
            'aleatoric_upper_bound': aleatoric_uncertainty_upper_bound,
        }

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        """
        Fits an NGBoost regressor to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
            y : DataFrame object or List or numpy array of target values (n) in Numeric format
            X_val : DataFrame object or List or numpy array of validation-set predictors (n x p) in Numeric format
            y_val : DataFrame object or List or numpy array of validation-set target values (n) in Numeric format

        Output:
            self : returns a fitted NGBRegressor object
        """
        # - Random seeds - #
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(0, 2**31 - 1, size=self.n_regressors)
        # ---------------- #

        match self.metadistribution_method:
            
            # --- Standard NGBoostRegressor - see https://arxiv.org/pdf/1910.03225 --- #
            case None | "None":
                super().__init__(
                    **self._core_ngboost_params()
                )
                return super().fit(X, y, X_val=X_val, Y_val=y_val, **kwargs)
            
            
            # --- Stochastic Gradient Langevin Boosting (SGLB) - see https://arxiv.org/pdf/2001.07248 --- #
            case "ensemble_SGLB":

                assert self.SGLB == True, "SGLB must be True for ensemble SGLB method."
                assert self.n_regressors > 1, "n_regressors must be greater than 1 for ensemble SGLB method."


                for i in range(self.n_regressors):
                    if self.verbose:
                        print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")

                    # Optionally combine with bagging
                    X_resampled, y_resampled = resample(
                        X, y,
                        replace=self.replace,
                        n_samples=int(len(X) * self.bagging_frac), 
                        random_state=seeds[i]  
                    )

                    model = NGBoost(
                        **self._core_ngboost_params(seed=seeds[i])
                    )
                    model.fit(X_resampled, y_resampled, **kwargs)
                    self.ensemble_models.append(model)


            # --- Stochastic Gradient Langevin Boosting (SGLB) - see https://arxiv.org/pdf/2006.10562 --- #
            case "virtual_SGLB":

                assert self.SGLB == True, "SGLB must be True for virtual SGLB method."
                assert self.n_regressors > 1, "n_regressors must be greater than 1 for virtual SGLB method."


                model = NGBoost(
                    **self._core_ngboost_params()
                )
                model.fit(X, y, **kwargs)


            # --- Stochastic Gradient Boosting (SGB) - see <INSERT PAPER LINK> --- #
            case "SGB":

                assert self.n_regressors > 1, "n_regressors must be greater than 1 for ensemble SGLB method."
                assert self.minibatch_frac < 1.0 or self.col_sample < 1.0, "minibatch_frac or col_sample must be less than 1 for SGB ensemble method."



                for i in range(self.n_regressors):            
                    # Optionally combine with bagging
                    X_resampled, y_resampled = resample(
                        X, y,
                        replace=self.replace,
                        n_samples=int(len(X) * self.bagging_frac), 
                        random_state=seeds[i]  
                    )

                    if self.verbose:
                        print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")

                    model = NGBoost(
                        **self._core_ngboost_params(seed=seeds[i])
                    )
                    model.fit(X_resampled, y_resampled, **kwargs)
                    self.ensemble_models.append(model)


            # --- Kernel Gradient Boosting (KGB) - see https://openreview.net/pdf?id=3VKiaagxw1S --- #
            case "KGB":
                raise NotImplementedError("KGB ensemble fitting logic is not yet implemented.")
            

            # --- Evidential Regression for Tree Boosting - see <INSERT PAPER LINK> --- #
            case "evidential_regression":

                assert hasattr(self.Dist, "pred_uncertainty") & (self.metadistribution_method == "evidential_regression"), "The distribution does not implement pred_uncertainty method."


                super().__init__(
                    **self._core_ngboost_params()
                )
                return super().fit(X, y, X_val=X_val, Y_val=y_val, **kwargs)

            # --- Building an ensemble using bootstrap aggregation (bagging) - see <INSERT PAPER LINK> --- #
            case "bagging":

                assert self.n_regressors > 1, "n_regressors must be greater than 1 for bagging method."
                assert (self.bagging_frac > 0) & (self.bagging_frac <= 1), "bagging_frac must be greater than 0 and at most 1 for bagging method."


                for i in range(self.n_regressors):
                    if self.verbose:
                        print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")

                    # Optionally combine with bagging
                    X_resampled, y_resampled = resample(
                        X, y,
                        replace=self.replace,
                        n_samples=int(len(X) * self.bagging_frac), 
                        random_state=seeds[i]  
                    )

                    model = NGBoost(
                        **self._core_ngboost_params(seed=seeds[i])
                    )
                    model.fit(X_resampled, y_resampled, **kwargs)
                    self.ensemble_models.append(model)
            case _:
                raise ValueError(f"Unknown metadistribution method: {self.metadistribution_method}")
            
            

    def _core_ngboost_params(self, seed: int = None):
        params = {
            "Dist": self.Dist,
            "Score": self.Score,
            "base_criterion": self.base_criterion,
            "splitter": self.splitter,
            "alpha": self.alpha,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "max_depth": self.max_depth,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "natural_gradient": self.natural_gradient,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "minibatch_frac": self.minibatch_frac,
            "col_sample": self.col_sample,
            "verbose": self.verbose,
            "verbose_eval": self.verbose_eval,
            "tol": self.tol,
            "random_state": seed if seed is not None else self.random_state,
            "validation_fraction": self.validation_fraction,
            "early_stopping_rounds": self.early_stopping_rounds,
            "SGLB": self.SGLB,
            "langevin_noise_scale": self.langevin_noise_scale,
            "epistemic_scaling": self.epistemic_scaling,
        }
        return params
    


class NGBClassifier(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost classification models.

    NGBRegressor is a wrapper for the generic NGBoost class that facilitates classification.
    Use this class if you want to predict an outcome that could take a discrete number of
    (unordered) values.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Bernoulli
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBClassifier object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Bernoulli,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):
        assert issubclass(
            Dist, ClassificationDistn
        ), f"{Dist.__name__} is not useable for classification."
        super().__init__(
            Dist,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )
        self._estimator_type = "classifier"

    def predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            Numpy array of the estimates of P(Y=k|X=x). Will have shape (n, K)
        """
        return self.pred_dist(X, max_iter=max_iter).class_probs()

    def staged_predict_proba(self, X, max_iter=None):
        """
        Probability prediction of Y at the points X=x at multiple boosting iterations

        Parameters:
            X        : numpy array of predictors (n x p)
            max_iter : largest number of boosting iterations to get the prediction for

        Output:
            A list of of the estimates of P(Y=k|X=x) of shape (n, K),
            one per boosting stage up to max_iter
        """
        return [
            dist.class_probs() for dist in self.staged_pred_dist(X, max_iter=max_iter)
        ]


class NGBSurvival(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost survival models.

    NGBSurvival is a wrapper for the generic NGBoost class that facilitates survival analysis.
    Use this class if you want to predict an outcome that could take an infinite number of
    (ordered) values, but right-censoring is present in the observed data.

     Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. LogNormal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
    Output:
        An NGBSurvival object that can be fit.
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=LogNormal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
    ):

        assert issubclass(
            Dist, RegressionDistn
        ), f"{Dist.__name__} is not useable for regression."
        if not hasattr(Dist, "censored_scores"):
            raise ValueError(
                f"The {Dist.__name__} distribution does not have any censored scores implemented."
            )

        SurvivalDistn = SurvivalDistnClass(Dist)

        # assert issubclass(Dist, RegressionDistn), f'{Dist.__name__} is not useable for survival.'
        super().__init__(
            SurvivalDistn,
            Score,
            Base,
            natural_gradient,
            n_estimators,
            learning_rate,
            minibatch_frac,
            col_sample,
            verbose,
            verbose_eval,
            tol,
            random_state,
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # Both of the below contain SurvivalDistn
        del state["Manifold"]
        state["_basedist"] = state["Dist"]._basedist
        del state["Dist"]
        return state

    def __setstate__(self, state_dict):
        # Recreate the object which could not be pickled
        state_dict["Dist"] = SurvivalDistnClass(state_dict["_basedist"])
        del state_dict["_basedist"]
        state_dict["Manifold"] = manifold(state_dict["Score"], state_dict["Dist"])
        self.__dict__ = state_dict

    # pylint: disable=too-many-positional-arguments
    def fit(self, X, T, E, X_val=None, T_val=None, E_val=None, **kwargs):
        """Fits an NGBoost survival model to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X                     : DataFrame object or List or
                                    numpy array of predictors (n x p) in Numeric format
            T                     : DataFrame object or List or
                                    numpy array of times to event or censoring (n) (floats).
            E                     : DataFrame object or List or
                                    numpy array of event indicators (n).
                                    E[i] = 1 <=> T[i] is the time of an event, else censoring time
            T_val                 : DataFrame object or List or
                                    validation-set times, in numeric format if any
            E_val                 : DataFrame object or List or
                                    validation-set event idicators, in numeric format if any
        """

        X = check_array(X, accept_sparse=True)

        if X_val is not None:
            X_val = check_array(X_val, accept_sparse=True)

        return super().fit(
            X,
            Y_from_censored(T, E),
            X_val=X_val,
            Y_val=Y_from_censored(T_val, E_val),
            **kwargs,
        )

class NGBEnsembleRegressor(NGBoost, BaseEstimator):
    """
    Constructor for NGBoost ensemble regression models.

    NGBEnsembleRegressor is a wrapper for the generic NGBoost class that facilitates an ensemble of NGBoost models for regression.
    Use this class if you want to predict an outcome that could take an
    infinite number of (ordered) values using an ensemble of NGBoost models.

    Parameters:
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor, e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to use in each boosting iteration
        col_sample        : the percent subsample of columns to use in each boosting iteration
        verbose           : flag indicating whether output should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility. See
                            https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
        validation_fraction: Proportion of training data to set
                             aside as validation data for early stopping.
        early_stopping_rounds:      The number of consecutive boosting iterations during which the
                                    loss has to increase before the algorithm stops early.
                                    Set to None to disable early stopping and validation.
                                    None enables running over the full data set.
        n_regressors    : the number of NGBoost regressors to fit in the ensemble
        ensemble_method : the method create the ensemble, options are
                            'stochastic_gradient_boosting' Creating an ensemble by fitting using a subsample of the data
                            'bagging' Creating an ensemble by fitting on random subsamples of the data
                            'SGLB' Creating an ensemble by fitting using a stochastic gradient boosting approach
                            'virtual_SGBL' Creating an ensemble by fitting using a virtual stochastic gradient boosting approach
        langevin_noise_scale: the scale of the Langevin noise to add during training (only used if ensemble_method='SGBL')
        bagging_frac    : the fraction of the data to use for each regressor in the ensemble when using bagging


    Output:
        An NGBRegressor object that can be fit.
    """
    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
        n_regressors=10,
        ensemble_method="stochastic_gradient_boosting",
        bagging_frac=0.8,
        langevin_noise_scale=1,
        ):
            assert issubclass(
                Dist, RegressionDistn
            ), f"{Dist.__name__} is not useable for regression."

            if not hasattr(
                Dist, "scores"
            ):  # user is trying to use a dist that only has censored scores implemented
                Dist = Dist.uncensor(Score)

            super().__init__(
                Dist,
                Score,
                Base,
                natural_gradient,
                n_estimators,
                learning_rate,
                minibatch_frac,
                col_sample,
                verbose,
                verbose_eval,
                tol,
                random_state,
                validation_fraction,
                early_stopping_rounds,
            )

            self._estimator_type = "regressor"
            self.n_regressors = n_regressors
            self.ensemble_method = ensemble_method
            self.models = []
            self.bagging_frac = bagging_frac
            self.langevin_noise_scale = langevin_noise_scale

    def fit(self, X, y, **kwargs):
        """
        Fits an NGBoost ensemble regressor to the data.
        For additional parameters see ngboost.NGboost.fit

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
            y : DataFrame object or List or numpy array of target values (n) in Numeric format

        Output:
            self : returns a list of fitted NGBRegressor models
        """

        # Train each weak learner with a subsample of the data (minibatch_frac) and randomize the minibatches between models
        if self.ensemble_method == "stochastic_gradient_boosting":
            rng = check_random_state(self.random_state)            
            for i in range(self.n_regressors):
                seed = rng.randint(0, 1e6)
                if self.verbose:
                    print(f"\n Fitting regressor [{i+1}/{self.n_regressors}] with seed {seed}")
                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=seed,
                    SGLB=False,
                )
                model.fit(X, y, **kwargs)
                self.models.append(model)

        # Train each model with a subsample of the data (bagging_frac) (Can be combined with a minibatch_frac for bagging + stochastic gradient boosting)
        elif self.ensemble_method == "bagging":
            for i in range(self.n_regressors):
                if self.verbose:
                    print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")
                # Use a subsample of the data for bagging
                X_resampled, y_resampled = resample(
                    X, y, 
                    replace=False,  # subsample without replacement
                    n_samples=int(len(X) * self.bagging_frac),  # fraction of data
                    random_state=self.random_state
                )
                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=self.random_state,
                    SGLB=False,  # Use standard NGBoost for bagging
                )
                model.fit(X_resampled, y_resampled, **kwargs)
                self.models.append(model)

        # Train each model using SGLB
        elif self.ensemble_method == "SGLB":
            for i in range(self.n_regressors):
                if self.verbose:
                    print(f"\n Fitting regressor [{i+1}/{self.n_regressors}]")

                model = NGBRegressor(
                    Dist=self.Dist,
                    Score=self.Score,
                    Base=self.Base,
                    natural_gradient=self.natural_gradient,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    minibatch_frac=self.minibatch_frac,
                    col_sample=self.col_sample,
                    verbose=self.verbose,
                    verbose_eval=self.verbose_eval,
                    tol=self.tol,
                    random_state=self.random_state,
                    SGLB=True, 
                    langevin_noise_scale=self.langevin_noise_scale,
                )
                model.fit(X, y, **kwargs)
                self.models.append(model)
        # Train an ensemble by truncating the SGLB model at various stages (Fit it as a single model, and use staged predictions)
        elif self.ensemble_method == "virtual_SGBL":
            model = NGBRegressor(
                Dist=self.Dist,
                Score=self.Score,
                Base=self.Base,
                natural_gradient=self.natural_gradient,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                minibatch_frac=self.minibatch_frac,
                col_sample=self.col_sample,
                verbose=self.verbose,
                verbose_eval=self.verbose_eval,
                tol=self.tol,
                random_state=self.random_state,
                SGLB=True, 
                langevin_noise_scale=self.langevin_noise_scale,
            )
            model.fit(X, y, **kwargs)
            self.models = [model]
        else:
            raise ValueError(f"Ensemble method {self.ensemble_method} not supported. Use 'stochastic_gradient_boosting' or 'SGLB' or 'bagging'.")
        return self
    
    def predict(self, X, average: bool = True):
        """
        Predicts the target values for the input data X using the ensemble of NGBoost models.
        For additional parameters see ngboost.NGboost.predict
        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format

        Output:
            Numpy array of predicted target values (n)
        """
        if average:
            predictions = [model.predict(X) for model in self.models]
            return np.mean(predictions, axis=0)
        else:
            return np.array([model.predict(X) for model in self.models]).T
        
    def pred_dist(self, X, max_iter=None):
        """
        Predicts the distribution of the target values for the input data X using the ensemble of NGBoost models.
        For additional parameters see ngboost.NGboost.pred_dist

        Parameters:
            X : DataFrame object or List or numpy array of predictors (n x p) in Numeric format
            max_iter : get the prediction at the specified number of boosting iterations

        Output:
            A list of predicted distributions for each model in the ensemble
        """
        return [model.pred_dist(X, max_iter=max_iter) for model in self.models]
    
    def pred_uncertainty(self, X, mode: str = 'bayesian_kl'):
        """
        Computes the Bayesian uncertainty disentanglement of the NGBoost ensemble (This only works with a Gaussian distribution).

        This method calculates both epistemic (model) and aleatoric (data) uncertainties for the predictions made by the ensemble of models. It aggregates predictions and variances from all models in the ensemble to provide a comprehensive uncertainty estimate for each prediction.

        Parameters:
            X (array-like): Input data for which to compute uncertainties

        Returns:
            Dict: A DataFrame containing the following columns for each training sample:
                - mean_prediction: The mean prediction across all models.
                - aleatoric_uncertainty: The mean predicted variance (aleatoric uncertainty) across all models.
                - epistemic_uncertainty: The variance of predictions across all models (epistemic uncertainty).#
                - predictive_uncertainty: The sum of aleatoric and epistemic uncertainties, representing the total predictive uncertainty.
        """
        if self.ensemble_method == "virtual_SGBL":
            staged_predictions = self.models[0].staged_pred_dist(X)
            # Extract n_regressors predictions after n_estimators/2 iterations
            indices = np.linspace(len(staged_predictions) // 2, len(staged_predictions) - 1, self.n_regressors, dtype=int)
            parameters = np.array([staged_predictions[i].params for i in indices])
            print(parameters.shape)
        else:
            parameters = np.array([model.pred_dist(X).params for model in self.models])
        
        
        mean_prediction = np.mean([param['loc'] for param in parameters], axis=0)
        if mode == 'bayesian_mean':    
            aleatoric_uncertainty = np.mean([param['scale'] for param in parameters], axis=0)
            epistemic_uncertainty = np.var([param['loc'] for param in parameters], axis=0)
        elif mode == 'levi_simple':
            aleatoric_uncertainty_lower_bound = np.min([param['scale'] for param in parameters], axis=0)
            aleatoric_uncertainty_upper_bound = np.max([param['scale'] for param in parameters], axis=0)
            
            # Extract all loc vectors
            locs = [param['loc'] for param in parameters]

            # Stack into array of shape (M, D) where M = # of models, D = dimension
            loc_matrix = np.stack(locs)  # shape: (M, D)

            # Compute pairwise L2 distances (or differences)
            pairwise_diff = loc_matrix[:, None, :] - loc_matrix[None, :, :]  # shape: (M, M, D)

            # Compute max difference across all model pairs and dimensions
            epistemic_uncertainty = np.max(np.abs(pairwise_diff), axis=(0, 1)) 
        elif mode == 'bayesian_kl':
            aleatoric_uncertainty = 0.5 * (1 + np.log(2*np.pi) + np.mean([np.log(param['scale']**2) for param in parameters], axis=0))
            mean_mu = np.mean([param['loc'] for param in parameters], axis=0)
            mean_sigma = np.mean([param['scale'] for param in parameters], axis=0)
            
            log_mean_sigma = np.mean(np.log([param['scale']**2 for param in parameters]), axis=0)
            mean_mu_deviation = np.mean([param['loc']**2 - mean_mu**2 for param in parameters], axis=0) / mean_sigma**2
            epistemic_uncertainty = 0.5 * (np.log(mean_sigma**2) - log_mean_sigma + mean_mu_deviation)
        elif mode == 'levi_kl':
            # The agent with the minimum entropy
            aleatoric_uncertainty_lower_bound = np.min([0.5 * np.log(2 * np.pi * np.e * param['scale']**2) for param in parameters], axis=0)
            # The agent with the maximum entropy
            aleatoric_uncertainty_upper_bound = np.max([0.5 * np.log(2 * np.pi * np.e * param['scale']**2) for param in parameters], axis=0)
            # Extract all loc vectors
            locs = [param['loc'] for param in parameters]
            variances = [param['scale'] for param in parameters]
            divergences = []
            for model_p in parameters:
                for model_q in parameters:
                    divergence = np.log(model_q['scale']) - np.log(model_p['scale']) + (model_p['scale']**2 + (model_p['loc'] - model_q['loc'])**2) / (2 * model_q['scale']**2) - 0.5
                    divergences.append(divergence)
            divergences = np.array(divergences)
            epistemic_uncertainty = np.max(divergences, axis=0)

        return {
            'mean': mean_prediction,
            'predictive': aleatoric_uncertainty + epistemic_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'predicitve_upper_bound': aleatoric_uncertainty_upper_bound + epistemic_uncertainty if 'aleatoric_uncertainty_upper_bound' in locals() else None,
            'predicitve_lower_bound': aleatoric_uncertainty_lower_bound + epistemic_uncertainty if 'aleatoric_uncertainty_lower_bound' in locals() else None,
            'aleatoric': aleatoric_uncertainty if 'aleatoric_uncertainty' in locals() else None,
            'epistemic': epistemic_uncertainty,
            'aleatoric_lower_bound': aleatoric_uncertainty_lower_bound if 'aleatoric_uncertainty_lower_bound' in locals() else None,
            'aleatoric_upper_bound': aleatoric_uncertainty_upper_bound if 'aleatoric_uncertainty_upper_bound' in locals() else None,
        }
