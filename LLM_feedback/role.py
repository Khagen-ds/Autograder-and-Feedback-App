# Importing abstractmethods from ABC
from abc import ABC, abstractmethod
#    Give the LLM a role
class Role(ABC):

    @abstractmethod
    def role_prompt(self):
        '''The model's given role to play as. Affects tone and approach to answering the tasks'''

class Teacher(Role):
    def role_prompt(self):
        return '''You are a teacher providing feedback to the student's essay. Focus on the student's understanding and adapt to the given essay'''

class llmAssistant(Role):
    def role_prompt(self):
        return '''You are a LLM assistant that helps teachers analyze essays and generate feedback to students. Let's think step by step in explaining the student's preformance'''

class Tutor(Role):
    def role_prompt(self):
        return '''You are a tutor that is providing feedback to the student's essay. The feedback should focus on areas that needs improvments, while aslo highlighting strenghts'''
