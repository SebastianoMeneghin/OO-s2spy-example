import lilio
import numpy  as np
import pandas as pd
import xarray as xr
import s2spy  as s2s
from s2spy import RGDR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from typing import List, Any, Iterator, Tuple


class CrossValidator:
    def __init__(self, k_fold_splits: int):
        self.k_fold_splits = k_fold_splits
        self.kfold = KFold(n_splits=self.k_fold_splits)

    def split_sets(self, x: Any, y: Any) -> Iterator[Tuple[Any, Any, Any, Any]]:
        cv = lilio.traintest.TrainTestSplit(self.kfold)
        # perform cross-validation
        return cv.split(x, y)
    
    
class DimensionalityReducer:    
    def __init__(self, target_intervals: int, lag: int, eps_km: float, alpha: float, min_area_km2: float):
        self.target_intervals = target_intervals
        self.lag              = lag
        self.eps_km           = eps_km
        self.alpha            = alpha
        self.min_area_km2     = min_area_km2
        # instanciate an RGDR object with the passed value
        self.rgdr = RGDR(
            target_intervals  = self.target_intervals,
            lag               = self.lag,
            eps_km            = self.eps_km,
            alpha             = self.alpha,
            min_area_km2      = self.min_area_km2
        )

    def transform(self, x_train: Any, y_train: Any, x_test: Any) -> Tuple[Any, Any]:
        # train the dimensionality reducer
        self.rgdr.fit(x_train, y_train)
        # transform the data into the new dimension
        clusters_train = self.rgdr.transform(x_train)
        clusters_test  = self.rgdr.transform(x_test)
        return clusters_train, clusters_test


class ModelTrainer:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def train(self, x: Any, y: Any) -> Any:
        # train model on data
        return self.model.fit(x, y)
    

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, y_true: Any, y_pred: Any) -> float:
        # calculate evaluation metric (e.g. RMSE)
        return mean_squared_error(y_true, y_pred)
    

class Predictor:
    def __init__(self, models: List[Any]):
        self.models = models

    def predict(self, x: Any) -> Any:
        # make predictions using trained models
        predictions = []
        for model in self.models:
            predictions.append[model.predict(x)]
        return predictions


class Main:
    def __init__(self, feature_values: Any, target_values: Any, target_intervals: int, lag: int, eps_km: int, alpha_reducer: float, min_area_km2: int, k_fold_splits: int, alpha_model: float):
        self.feature_values   = precursor_field_sel
        self.target_values    = target_series_sel
        self.target_intervals = target_intervals
        self.lag              = lag
        self.eps_km           = eps_km
        self.alpha_reducer    = alpha_reducer
        self.min_area_km2     = min_area_km2
        self.k_fold_splits    = k_fold_splits
        self.alpha_model      = alpha_model
        

    def run(self):
        # instanciate dimensionality reducer
        dim_reducer   = DimensionalityReducer(self.target_intervals, self.lag, self.eps_km, self.alpha_reducer, self.min_area_km2)
        
        # instanciate model trainer
        model_trainer = ModelTrainer(self.alpha_model)

        # instanciate cross validator
        cv            = CrossValidator(self.k_fold_splits)

        # instanciate evaluator
        evaluator     = Evaluator()

        # create lists for saving models, predictions, and related info
        models            = []
        predictions       = []
        rmse_train        = []
        rmse_test         = []
        train_test_splits = []

        # cross-validation loop
        for x_train, x_test, y_train, y_test in cv.split(self.feature_values, self.target_values):
            # log train/test splits with anchor years
            train_test_splits.append({
                "train": x_train.anchor_year.values,
                "test":  x_test.anchor_year.values,
            })

            # fit dimensionality reduction operator RGDR and transform test and train data
            clusters_train, clusters_test = dim_reducer.transform(x_train, x_test)

            # train model
            model = model_trainer.train(clusters_train.isel(i_interval=0), y_train.sel(i_interval=1))
        
            # save model
            models.append(model)

            # predict and save results
            prediction = model.predict(clusters_test.isel(i_interval=0))
            predictions.append(prediction)

            # calculate and save rmse on train and on test sets
            rmse_train.append(evaluator.evaluate(y_train.sel(i_interval=1), model.predict(clusters_train.isel(i_interval=0))))
            rmse_test.append( evaluator.evaluate(y_test.sel(i_interval=1),  prediction))

        ''' 
        once selected the models, in future implementation:
        # create predictor
        predictor = Predictor(models)
        '''

        return models, rmse_train, rmse_test, train_test_splits


if __name__ == "__main__":
    precursor_field_sel = None # add your own precursor field selection
    target_series_sel   = None # add your own target field selection

    main = Main(precursor_field_sel, target_series_sel, target_series_sel, target_intervals=1, lag=2, eps_km=600, alpha_reducer=0.05, min_area_km2=0, k_fold_splits=5, alpha_model=1.0)
    models, rmse_train, rmse_test, train_test_splits = main.run()