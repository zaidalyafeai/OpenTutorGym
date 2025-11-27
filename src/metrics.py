import json

class Metric:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            score = 1
        elif response.lower() == 'to some extent':
            score = 0.5
        else:
            score = 0
        return {self.metric_name: score}

class MistakeIdentification(Metric):
    def __init__(self):
        super().__init__('mistake_identification')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Has the tutor identified a mistake in the student’s response?
        Return:
        Yes, To some extent, No
        """

class AnswerReveal(Metric):
    def __init__(self):
        super().__init__('answer_reveal')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor reveal the final answer to the student?
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return {self.metric_name: 0}
        else:
            return {self.metric_name: 1}

class ProvidingGuidance(Metric):
    def __init__(self):
        super().__init__('providing_guidance')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor offer correct and relevant guidance?
        Return:
        Yes, To some extent, No
        """
            
class Helpfulness(Metric):
    def __init__(self):
        super().__init__('helpfulness')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the tutor’s response helpful to the student?
        Return:
        Yes, To some extent, No
        """

class Creativity(Metric):
    def __init__(self):
        super().__init__('creativity')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the tutor’s response creative and engaging?
        Return:
        Yes, To some extent, No
        """

class Clarity(Metric):
    def __init__(self):
        super().__init__('clarity')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the tutor’s response clear and easy to understand?
        Return:
        Yes, To some extent, No
        """

class SelfCorrection(Metric):
    def __init__(self):
        super().__init__('self_correction')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor encourage the student to correct their own mistakes?
        Return:
        Yes, To some extent, No
        """

class OverLoad(Metric):
    def __init__(self):
        super().__init__('overload')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor provide too much information?
        Return:
        Yes, To some extent, No
        """

class Feedback(Metric):
    def __init__(self):
        super().__init__('feedback')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the feedback of the tutor response is helpful and specific?
        Return:
        Yes, To some extent, No
        """

class Support(Metric):
    def __init__(self):
        super().__init__('support')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the tutor’s response supportive and encouraging?
        Return:
        Yes, To some extent, No
        """


class TutorTone(Metric):
    def __init__(self):
        super().__init__('tutor_tone')
        self.system_prompt = """You are given a conversation. 
        What is the tone of the tutor’s response?
        Return:
        Encouraging, Neutral, Offensive
        """
    
    def calculate_score(self, response):
        if response.lower() == 'encouraging':
            return {self.metric_name: 1}
        elif response.lower() == 'neutral':
            return {self.metric_name: 0.5}
        else:
            return {self.metric_name: 0}

class CorrectAnswer(Metric):
    def __init__(self):
        super().__init__('correct_answer')
        self.system_prompt = """You are given a conversation between tutor and student. 
        you need to judge if the provided answer by the student is correct.
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return {self.metric_name: 1}
        else:
            return {self.metric_name: 0}

class Coherence(Metric):
    def __init__(self):
        super().__init__('coherence')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Is the conversation coherent and easy to follow?
        Return:
        Yes, To some extent, No
        """

class Actionability(Metric):
    def __init__(self):
        super().__init__('actionability')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor provide actionability of the feedback through nudges, questions, and hints?
        Return:
        Yes, To some extent, No
        """

class Informativeness(Metric):
    def __init__(self):
        super().__init__('informativeness')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor provide relevant information to the student in each turn?
        Return:
        Yes, To some extent, No
        """

class Care(Metric):
    def __init__(self):
        super().__init__('care')
        self.system_prompt = """You are given a conversation between tutor and student. 
        Does the tutor show care and empathy towards the student?
        Return:
        Yes, To some extent, No
        """

all_metrics = {
    'mistake_identification': MistakeIdentification(),
    'answer_reveal': AnswerReveal(),
    'providing_guidance': ProvidingGuidance(),
    'tutor_tone': TutorTone(),
    'correct_answer': CorrectAnswer(),
    'support': Support(),
    'feedback': Feedback(),
    'clarity': Clarity(),
    'helpfulness': Helpfulness(),
    'creativity': Creativity(),
    'self_correction': SelfCorrection(),
    'overload': OverLoad(),
    'actionability': Actionability(),
    'informativeness': Informativeness(),
    'care': Care()
}

class AggregatedMetric(Metric):
    def __init__(self, metrics: list[str]):
        super().__init__('aggregated_metric')
        self.metrics = metrics
        self.system_prompt = ""
        output_json = ""
        for i, metric_name in enumerate(metrics):
            self.system_prompt += all_metrics[metric_name].system_prompt
            comma = ", " if i < len(metrics) - 1 else ""
            output_json += f'"{metric_name}": answer{comma}\n'
        self.system_prompt += f"""
        return a JSON object with the following keys:
        {output_json}
        """
    def fix_json(self, response: str):
        response = response.replace("```", "")
        response = response.replace("json", "")
        return response

    def calculate_score(self, response: str):
        try:
            fixed_response = self.fix_json(response)
            response = json.loads(fixed_response)
            return {metric_name: all_metrics[metric_name].calculate_score(response[metric_name])[metric_name] for metric_name in self.metrics}
        except Exception as e:
            print(f"Error evaluating conversation: {str(e)}")
            return {metric_name: 0.0 for metric_name in self.metrics}
    
def load_metric(metric_names: list[str]):
    if len(metric_names) > 1:
        return AggregatedMetric(metric_names)
    return all_metrics[metric_names[0]]        