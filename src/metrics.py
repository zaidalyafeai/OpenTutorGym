class Metric:
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        elif response.lower() == 'to some extent':
            return 0.5
        else:
            return 0

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
        Does the tutor reveal the final answer?
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 0
        else:
            return 1

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
            return 1
        elif response.lower() == 'neutral':
            return 0.5
        else:
            return 0

class CorrectAnswer(Metric):
    def __init__(self):
        super().__init__('correct_answer')
        self.system_prompt = """You are given a conversation between tutor and student. 
        you need to judge if the provided answer matches the gold answer.
        Return:
        Yes, No
        """
    
    def calculate_score(self, response):
        if response.lower() == 'yes':
            return 1
        else:
            return 0

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
    elif metric_name == 'support':
        return Support()
    elif metric_name == 'feedback':
        return Feedback()
    elif metric_name == 'clarity':
        return Clarity()
    elif metric_name == 'helpfulness':
        return Helpfulness()
    elif metric_name == 'creativity':
        return Creativity()
    elif metric_name == 'self_correction':
        return SelfCorrection()
    elif metric_name == 'overload':
        return OverLoad()
    elif metric_name == 'actionability':
        return Actionability()
    elif metric_name == 'informativeness':
        return Informativeness()
    elif metric_name == 'care':
        return Care()
    else:
        raise ValueError(f"Invalid metric: {metric_name}")
        