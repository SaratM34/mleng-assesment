import pandas as pd
import numpy as np
import re
import os

import requests
import onnxruntime as rt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


def clean_transform_title(job_title):
    """Clean and transform job title. Remove punctuations, special characters,
    multiple spaces etc.
    """
    if not isinstance(job_title, str):
        return ''
    new_job_title = job_title.lower()
    special_characters = re.compile('[^ a-zA-Z]')
    new_job_title = re.sub(special_characters, ' ', new_job_title)
    extra_spaces = re.compile(r'\s+')
    new_job_title = re.sub(extra_spaces, ' ', new_job_title)

    return new_job_title


class SeniorityModel:
    """Job seniority model class. Contains attributes to fit, predict,
    save and load the job seniority model.
    """

    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.api_url = "https://api.salesloft.com/v2/people.json"
        self.api_key = os.getenv('API_KEY')

    def _check_for_array(self, variable):
        if not isinstance(variable, (list, tuple, np.ndarray)):
            raise TypeError("variable should be of type list or numpy array.")
        return

    def _data_check(self, job_titles, job_seniorities):
        self._check_for_array(job_titles)
        self._check_for_array(job_seniorities)

        if len(job_titles) != len(job_seniorities):
            raise IndexError("job_titles and job_seniorities must be of the same length.")

        return

    def fit(self, job_titles, job_seniorities):
        """Fits the model to predict job seniority from job titles.
        Note that job_titles and job_seniorities must be of the same length.

        Parameters
        ----------
        job_titles: numpy array or list of strings representing job titles
        job_seniorities: numpy array or list of strings representing job seniorities
        """
        self._data_check(job_titles, job_seniorities)

        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])

        self.vectorizer = CountVectorizer(ngram_range=(1, 2))
        self.model = Pipeline([('vectorizer', self.vectorizer), ('svc', LinearSVC())])
        self.model.fit(cleaned_job_titles, job_seniorities)
        return

    def predict(self, job_titles):

        if self.model is None:
            raise ValueError('Model object is None. Model should be loaded before calling predict function. Call load function before predict.')

        self._check_for_array(job_titles)
        cleaned_job_titles = np.array([clean_transform_title(jt) for jt in job_titles])
        cleaned_job_titles = cleaned_job_titles.reshape((-1, 1))

        predictions = self.model.run(None, {'strfeat': cleaned_job_titles})[0]
        return predictions

    def _create_input(self, input):
        """Creates the input for the model from the response json."""
        ids = []
        job_titles = []
        for item in input['data']:
            ids.append(item['id'])
            job_titles.append(item['title'])
        return ids, job_titles

    def predict_salesloft_team(self):
        """Calls the salesloft API to recieve the list of all people and the run the job titles through the model."""
        response = requests.get(self.api_url,
                                headers={"Accept": "application/json", "Authorization": f"Bearer {self.api_key}"})
        data = response.json()
        ids, job_titles = self._create_input(data)
        predictions = self.predict(job_titles)
        return list(zip(ids, predictions))

    def save(self, file_name):
        """Saves the model file in an language-agnostic format(onnx) at the specified location."""
        if self.model is None:
            raise ValueError('Model object is None. Model should be trained before saving')

        initial_type = [('strfeat', StringTensorType([None, 1]))]
        onx = convert_sklearn(self.model, "vectorizer", initial_types=initial_type, target_opset=12)
        with open(file_name, "wb") as f:
            f.write(onx.SerializeToString())

    def load(self, file_name):
        """Loads the saved model from a given location."""
        if not os.path.isfile(file_name):
            raise FileNotFoundError(f'{file_name} does not exist.')

        self.model = rt.InferenceSession(file_name)


if __name__ == '__main__':
    data = pd.read_csv('../data/title_data_for_model_fitting.csv')
    job_titles = data['job_title'].values
    job_seniority = data['job_seniority'].values
    s = SeniorityModel()
    s.fit(job_titles, job_seniority)
    s.save('../models/seniority_model_v1.onnx')
    s.load('../models/seniority_model_v1.onnx')
    print(s.predict(job_titles))
    print(s.predict_salesloft_team())
