import unittest
import numpy as np
from src.main import CrossValidator, DimensionalityReducer, ModelTrainer, Evaluator, Predictor, Main
from sklearn.linear_model import Ridge

class TestCrossValidator(unittest.TestCase):
    def test_splits_sets(self):
        cv = CrossValidator(k_fold_splits=5)
        feature_values = np.random.rand(100, 10)
        target_values = np.random.rand(100)
        # cast to list for assure existance of method "len"
        splits = list(cv.split(feature_values, target_values))
        self.assertEqual(len(splits), 5)


class TestDimensionalityReducer(unittest.TestCase):
    def test_transform(self):
        dim_reducer = DimensionalityReducer(
            target_intervals=1,
            lag=2,
            eps_km=600,
            alpha=0.05,
            min_area_km2=0
        )
        feature_values = np.random.rand(100, 10)
        transformed_data = dim_reducer.transform(feature_values)
        self.assertIsInstance(transformed_data, np.ndarray)


class TestModelTrainer(unittest.TestCase):
    def test_train(self):
        model_trainer = ModelTrainer(alpha=1.0)
        feature_values = np.random.rand(100, 10)
        target_values = np.random.rand(100)
        model = model_trainer.train(feature_values, target_values)
        self.assertIsInstance(model, Ridge)


class TestEvaluator(unittest.TestCase):
    def test_evaluate(self):
        evaluator = Evaluator()
        y_true = np.random.rand(100)
        y_pred = np.random.rand(100)
        rmse = evaluator.evaluate(y_true, y_pred)
        self.assertIsInstance(rmse, float)


'''
Implement the rest of the tests
'''