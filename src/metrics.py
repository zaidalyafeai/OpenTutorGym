class Metric:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name

class MistakeIdentification(Metric):
    def __init__(self):
        super().__init__('mistake_identification')
        self.system_prompt = """You are given a conversation. 
        Has the tutor identified a mistake in the student’s response?
        Return:
        Yes, To some extent, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        elif response.lower() == 'to some extent':
            return 0.5
        else:
            return 0

class AnswerReveal(Metric):
    def __init__(self):
        super().__init__('answer_reveal')
        self.system_prompt = """You are given a conversation. 
        Does the tutor reveal the final answer?
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        else:
            return 0

class ProvidingGuidance(Metric):
    def __init__(self):
        super().__init__('providing_guidance')
        self.system_prompt = """You are given a conversation. 
        Does the tutor offer correct and relevant guidance?
        Return:
        Yes, To some extent, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        elif response.lower() == 'to some extent':
            return 0.5
        else:
            return 0

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
            return 1
        elif response.lower() == 'neutral':
            return 0.5
        else:
            return 0

class CorrectAnswer(Metric):
    def __init__(self):
        super().__init__('correct_answer')
        self.system_prompt = """You are given a conversation, you need to judge 
        if the provided answer matches the gold answer.
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        else:
            return 0
            

def load_metric(metric_name: str) -> Metric:
    if metric_name == 'mistake_identification':
        return MistakeIdentification()
    elif metric_name == 'answer_reveal':
        return AnswerReveal()
    elif metric_name == 'providing_guidance':
        return ProvidingGuidance()
    elif metric_name == 'tutor_tone':
        return TutorTone()
    elif metric_name == 'correct_answer':
        return CorrectAnswer()
    else:
        raise ValueError(f"Invalid metric: {metric_name}")
        