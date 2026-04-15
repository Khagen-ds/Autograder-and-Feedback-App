# Importing abstractmethods from ABC
from abc import ABC, abstractmethod

#     Give the LLM areas it should focus on in its feedback
class OutcomeFocus(ABC):

    @abstractmethod
    def get_instructions(self):
        '''
        Applying what areas the LLM should focus to give feedback on, providing the LLM with a clear structure to get quality feedback.
        Content class gives feedback on the content of the essay only.
        Grammar class gives feedback on grammar from the essay only
        '''

class Content(OutcomeFocus):
    def get_instructions(self):
        return '''

        Content feedback instructions:
        Give feedback only on:
        - The students ideas
        - How well the assignment is answered
        
        Rules:
        - Do NOT give feedback on grammar
        - Do NOT rewrite sentences
        - Provide guidance, NOT finished solutions
        
        Output format:
        Strengths: (3 bullet points):
            - Each strenght must refrence a specific part from the student's essay
            - Each strenght must meet a requirement in the assignment

        Improvments (3 bullet points):
            - Identify missing or weak elements
            - Explain why the missing or weak element matters
            - Specify what are missing or weak in the essay compared to the assignment

        Concrete steps (2 bullet points):
            - Each step must be an action the student can apply
            - Each step must link to an improvement

        Summarize (5 bullet points):
            - Content quality
            - Use of relevant evidence
            - Explanation depth 
            - Cohesion and structure
            - Quality of arguments and explenations

        Follow the output format exactly.
        Do not add extra sections.
                
        '''

class Grammar(OutcomeFocus):
    def get_instructions(self):
        return '''

        Grammar feedback instructions:
        Give feedback only on:
        - Grammar mistakes
        - Sentence structure
        - Clarity

        Rules:
        - Do NOT evaluate content
        - Do NOT change the meaning of sentences

        Output format:
        Grammar:
        - List incorrect phrases and words. Example essay sentence: "The dog caught a had ball but got noting to land on causing it to be injured"
            The list should look like this: 
                - 'had ball' - 'hard ball' 
                - 'noting' - 'nothing' 
        Structure:
        - Identify unclear sentences
        - Identify very long sentences
        - Provide suggestions for improvements, NOT the direct answer

        Tips to avoid common mistakes:
        - Provide writing advice
        - Provide an example of a clear sentence structure.
        

        Follow the output format exactly.
        Do not add extra sections.        
        '''
