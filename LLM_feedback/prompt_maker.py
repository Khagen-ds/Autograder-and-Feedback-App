from LLM_feedback.role import Teacher, Tutor, llmAssistant
from LLM_feedback.focus import Content, Grammar
from LLM_feedback.rules import CurrentRules


class FeedbackPrompt:
    """
    Combines all components into a final feedback prompt
    """

    def __init__(self, score, role, outcome, essay, rules):
        self.score = score
        self.role = role
        self.outcome = outcome
        self.essay = essay
        self.rules = rules

    def generate_prompt(self):

        prompt = f"""
        ### Role
        {self.role.role_prompt()}
            
        ### Rules
        {self.rules.get_rules()}
            
        ### Score
        Score: {self.score.score}
            
        {self.score.score_description()}
            
        ### Task
        {self.outcome.get_instructions()}
            
        ### Assignment
        {self.essay.assignment}
            
        ### Article
        {self.essay.assignment_article}
            
        ### Student Essay
        {self.essay.essay}
        """

            
        return prompt.strip()


def create_feedback_prompt(score, role_type, outcome_type, essay):

    roles = {
        "teacher": Teacher(),
        "assistant": llmAssistant(),
        "tutor": Tutor()
    }

    outcomes = {
        "content": Content(),
        "grammar": Grammar()
    }

    rules = CurrentRules()

    return FeedbackPrompt(
        score=score,
        role=roles[role_type],
        outcome=outcomes[outcome_type],
        essay=essay,
        rules=rules
    )
