import unittest
import pandas as pd
import numpy as np
import os

from exercise.model import SeniorityModel, clean_transform_title

class TestModel(unittest.TestCase):

    data = pd.read_csv('data/title_data_for_model_fitting.csv')
    job_titles = data['job_title'].values
    job_seniority = data['job_seniority'].values

    def test_model_save(self):
        """
        Test the save function
        """
        s = SeniorityModel()

        s.fit(self.job_titles, self.job_seniority)
        s.save('models/seniority_model_v1_test.onnx')
        self.assertTrue(os.path.isfile('models/seniority_model_v1_test.onnx'))

    def test_model_load(self):
        """
        Test the Model Load function
        """
        s = SeniorityModel()
        s.load('models/seniority_model_v1_test.onnx')
        self.assertNotEqual(s.model, None)

    def test_predictions(self):
        """
        Test that predictions from persisted model and hydrated model are same
        """

        s = SeniorityModel()

        s.fit(self.job_titles, self.job_seniority)
        s.save('models/seniority_model_v1_test.onnx')

        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in self.job_titles])
        original_predictions = s.model.predict(cleaned_job_titles)

        s.load('models/seniority_model_v1_test.onnx')
        persisted_predictions = s.predict(self.job_titles)

        diff = 0
        for v1, v2 in zip(original_predictions, persisted_predictions):
            if v1 != v2:
                diff += 1

        self.assertEqual(diff, 0)

if __name__ == '__main__':
    unittest.main()