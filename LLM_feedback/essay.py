# Importing abstractmethods from ABC
from abc import ABC, abstractmethod

class Essay:
    '''The provided essay and assignment'''

    def __init__(self, essay = None, assignment = None, assignment_article = None):
        self._essay = essay
        self._assignment = assignment
        self._assignment_article = assignment_article 

    @property
    def essay(self):
        return self._essay

    @property
    def assignment(self):
        return self._assignment

    @essay.setter
    def essay(self, essay):
        self._essay = essay

    @property
    def assignment_article(self):
        return self._assignment_article

    @assignment_article.setter
    def assignment_article(self, assignment_article):
        self._assignment_article = assignment_article


    @assignment.setter
    def assignment(self, assignment):
        self._assignment = assignment

    def __str__(self):
        return f' Essay: \n {self._essay}\n Assignment: \n{self._assignment}\n Article:\n {self._assignment_article}'

    def get_random_essay(df):
        '''Function for the app to select a random essay from the testing dataset (data the model has not seen)'''
        row = df.sample(1).iloc[0]
        return Essay(
            essay=row["full_text"],
            assignment=row["assignment"],
            assignment_article=row["source_text_1"]
        )