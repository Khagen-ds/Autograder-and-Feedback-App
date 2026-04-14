# Importing abstractmethods from ABC
from abc import ABC, abstractmethod

# Get a score from the classifier model + confidence and actual score
class Score(ABC):
    '''The Autograder model's score (1-6) And explenation of the grade given by the ASAP 2.0 corpus (Score description)'''
    def __init__(self, score: int):
        self._score = score 

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, val):
        self._score = val

    @score.setter
    def score(self, val):
        if not (1 <= val <= 6):
            raise ValueError("Score must be between 1 and 6")
        self._score = val
        
    def score_description(self):
        descript = {
            1 : '''
                An essay in this category demonstrates very little or no mastery, and is severely flawed by ONE OR MORE of the 
                following weaknesses: develops no viable point of view on the issue, or provides little or no evidence to support
                its position; the essay is disorganized or unfocused, resulting in a disjointed or incoherent essay; 
                the essay displays fundamental errors in vocabulary and/or demonstrates severe flaws in sentence structure; 
                the essay contains pervasive errors in grammar, usage, or mechanics that persistently interfere with meaning.
                ''',
                
            2 : '''
                An essay in this category demonstrates little mastery, and is flawed by ONE OR MORE of the following weaknesses: 
                Develops a point of view on the issue that is vague or seriously limited, and demonstrates weak critical thinking; the essay provides 
                inappropriate or insufficient examples, reasons, or other evidence taken from the source text to support its position; the essay is poorly 
                organized and/or focused, or demonstrates serious problems with coherence or progression of ideas; the essay displays very little 
                facility in the use of language, using very limited vocabulary or incorrect word choice and/or demonstrates frequent problems in 
                sentence structure; the essay contains errors in grammar, usage, and mechanics so serious that meaning is somewhat obscured.
                ''',
            
            3 : '''
                An essay in this category demonstrates developing mastery, and is marked by ONE OR MORE of the following 
                weaknesses: develops a point of view on the issue, demonstrating some critical thinking, but may do so inconsistently or use inadequate 
                examples, reasons, or other evidence taken from the source texts to support its position; the essay is limited in its organization or focus, 
                or may demonstrate some lapses in coherence or progression of ideas displays; the essay may demonstrate facility in the use of language, 
                but sometimes uses weak vocabulary or inappropriate word choice and/or lacks variety or demonstrates problems in sentence 
                structure; the essay may contain an accumulation of errors in grammar, usage, and mechanics.
                ''',
            
            4 : '''
                An essay in this category demonstrates adequate mastery, although it will have lapses in quality. A typical essay 
                develops a point of view on the issue and demonstrates competent critical thinking; the essay using adequate examples, reasons, and 
                other evidence taken from the source text(s) to support its position; the essay is generally organized and focused, demonstrating some 
                coherence and progression of ideas exhibits adequate; the essay may demonstrate inconsistent facility in the use of language, using 
                generally appropriate vocabulary demonstrates some variety in sentence structure; the essay may have some errors in grammar, usage, 
                and mechanics.
                ''',
            
            5 : '''
                An essay in this category demonstrates reasonably consistent mastery, although it will have occasional errors or lapses in quality. 
                A typical essay effectively develops a point of view on the issue and demonstrates strong critical thinking; 
                the essay generally using appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; 
                the essay is well organized and focused, demonstrating coherence and progression of ideas; 
                the essay exhibits facility in the use of language, using appropriate vocabulary demonstrates variety in sentence structure; 
                the essay is generally free of most errors in grammar, usage, and mechanics.
                ''',
            
            6 : '''
                An essay in this category demonstrates clear and consistent mastery, although it may have a few minor errors. A 
                typical essay effectively and insightfully develops a point of view on the issue and demonstrates outstanding critical thinking; the essay 
                uses clearly appropriate examples, reasons, and other evidence taken from the source text(s) to support its position; the essay is well 
                organized and clearly focused, demonstrating clear coherence and smooth progression of ideas; the essay exhibits skillful use of 
                language, using a varied, accurate, and apt vocabulary and demonstrates meaningful variety in sentence structure; the essay is free of 
                most errors in grammar, usage, and mechanics.
                '''
        }

        
        return descript.get(self._score, "Unknown score")
        


