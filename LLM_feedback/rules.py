# Importing abstractmethods from ABC
from abc import ABC, abstractmethod

class Rules(ABC):
    '''
    Rules for the LLM to follow. Very imnportant for consistentcy. 
    Use: NOT and MUST and try to make the rule clear and concise
    '''
    @abstractmethod
    def get_rules(self):
        '''What rules the LLM must follow'''

class CurrentRules(Rules):

    def get_rules(self):
        return '''
        General Rules:
        - Do NOT discuss or meantion the score. 
        - Do NOT rewrite the essay
        - Provide guidance, NOT full solutions
        - Think carefully about the essay before generating feedback
        - Ensure all feedback is consistent with the score
        - Feedback must reflect the given score level
        - Higher scores should focus on strengths
        - Lower scores should focus on improvements
        - Base feedback on assignment requirements
        - Follow the output format excatly
        - Do NOT add extra sections or explanations outside the format
        - Do NOT skip any section
        - Only reference content that exist in the sutdent's essay
        - Do NOT assume missing information
        - Do NOT repeat the same feedback in multiple bullet points
        - Do NOT include any labels such as "Referance:", "Evidence:", or "Qoute:"
        - Do NOT qoute sentences from the essay verbatim
        
        '''