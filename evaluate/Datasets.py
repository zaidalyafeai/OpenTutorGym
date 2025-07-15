import json
import requests
from PIL import Image
import zipfile
import os
import hashlib
import datasets
class Problem:
    def __init__(self, text = '', img = None):
        self.text = text
        if img is not None:
            if os.path.exists(img):
                self.img = Image.open(img).convert('RGB')
            else:
                self.img = None
                # print(f"Image {img} not found")
        else:
            self.img = None

    def __str__(self):
        return f"Problem: {self.text}\n"

class Conversation:
    def __init__(self, problem:Problem = None, solution = '', topic = '', dialouge = []):
        if problem is None:
            self.problem = Problem()
        else:
            self.problem = problem
        self.solution = solution
        self.topic = topic
        self.dialouge = [{"role": item['role'], "content": item['content']} for item in dialouge]
    
    def __str__(self):
        out = ""
        out += f"Question: {self.problem.text}\n"
        out += f"Solution: {self.solution}\n"
        out += f"Topic: {self.topic}\n"
        out += f"Dialouge: \n"
        for turn in self.dialouge:
            out += f"{turn['role']}: {turn['content']}\n"
        out += f"--------------------------------\n"
        return out

def hash_question(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()

def load_dataset(dataset_name: str):
    if dataset_name == "gsm8k":
        return GSM8K()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

class GSM8K(Dataset):
    def __init__(self, split = "test"):
        self.dataset_name = "gsm8k"
        self.split = split
        self.subset = "main"
        self.hf_name = "openai/gsm8k"
        self.dataset = datasets.load_dataset(self.hf_name, self.subset, streaming = True)[self.split]

    
    def process_dataset(self):
        self.dataset = self.dataset.map(lambda x: {"question": x['question'], "answer": x['answer'].split("####")[1].strip()})
        self.dataset = self.dataset.map(lambda x: {"id": hash_question(x["question"])})
        return self.dataset
    
    
class DialougeDataset:
    def __init__(self, conversations = [], subset = 'train'):
        self.subset = subset
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]

    def get_dialouge(self, flatten = False):
        conversations = []
        for conversation in self.conversations:
            dialouge = []
            for turn in conversation.dialouge:
                dialouge.append({"role": 'user' if turn['role'] == 'student' else 'assistant', "content": turn['content']})
            conversations.append(dialouge)
        return conversations

class StepVerify(DialougeDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = 'https://raw.githubusercontent.com/eth-lre/verify-then-generate/refs/heads/main/dataset/dataset.json'
        self.conversations = []
        for item in json.loads(requests.get(path).text):
            dialouge = []
            for turn in item['dialog_history']:
                role = "student" if turn['user'] == "Student" else "tutor"
                dialouge.append({"role": role , "content": turn['text']})
            dialouge.append({"role": "student" , "content": item['student_correct_response']})
            self.conversations.append(Conversation(problem = Problem(text = item['problem']), solution = item['reference_solution'], topic = item['topic'], dialouge = dialouge))

class MathDial(DialougeDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        """
        "Teacher: (probing)Steven, If you had 4 of something and tripled that amount, how much would you have?|EOM|Steven: I would have 12 of something.|EOM|Teacher: (probing)So if Nancy triples the 18 cubic feet of water, how much would she have?|EOM|Steven: She would have 54 cubic feet of water.|EOM|Teacher: (generic)Exactly correct!"}
        """
        path = 'https://raw.githubusercontent.com/eth-nlped/mathdial/refs/heads/main/data/train.jsonl'
        self.conversations = []
        lines = requests.get(path).text.split("\n")
        for line in lines:
            if line == "":
                continue
            item = json.loads(line.strip())
            dialouge = []
            
            for turn in item['conversation'].split("|EOM|"):
                if len(turn.split(":")) <= 1:
                    raise Exception("stop")
                elif len(turn.split(":")) > 2:
                    entity = turn.split(":")[0].strip()
                    content = ":".join(turn.split(":")[1:]).strip()
                else:
                    entity = turn.split(":")[0].strip()
                    content = turn.split(":")[1].strip()
                
                if content.startswith("("):
                    content = content.split(")")[-1].strip()
                if 'Teacher' in entity:
                    dialouge.append({"role": "tutor" , "content": content})
                else:
                    dialouge.append({"role": "student" , "content": content})
            dialouge.append({"role": "student" , "content": item['student_incorrect_solution']})
            self.conversations.append(Conversation(problem = Problem(text = item['question']), solution = item['ground_truth'], topic = item['student_profile'], dialouge = dialouge))

class Bridge(DialougeDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        if self.subset == 'train':
            path = 'https://raw.githubusercontent.com/rosewang2008/bridge/refs/heads/main/dataset/train.json'
        elif self.subset == 'dev':
            path = 'https://raw.githubusercontent.com/rosewang2008/bridge/refs/heads/main/dataset/validation.json'
        elif self.subset == 'test':
            path = 'https://raw.githubusercontent.com/rosewang2008/bridge/refs/heads/main/dataset/test.json'
        else:
            raise ValueError(f"Dataset {self.subset} not found")
        self.conversations = []
        lines = json.loads(requests.get(path).text)
        for item in lines:
            dialouge = []
            for turn in item['c_h']:
                dialouge.append({"role": turn['user'], "content": turn['text']})
            if 'c_revision' in item:
                for turn in item['c_revision']:
                    dialouge.append({"role": turn['user'], "content": turn['text']})
            self.conversations.append(Conversation(topic = item['lesson_topic'], dialouge = dialouge))

class Cima(DialougeDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = "https://raw.githubusercontent.com/kstats/CIMA/refs/heads/master/dataset.json"
        self.conversations = []
        lines = json.loads(requests.get(path).text)["prepDataset"]
        # it is faster to download all the images in .cache/cima/images
        if not os.path.exists('.cache/cima/images'):
            os.makedirs('.cache/cima/images')
        
        # download the directory https://github.com/kstats/CIMA/tree/master/pictures as zip then unzip it
        zip_path = '.cache/cima/pictures.zip'
        if not os.path.exists(zip_path):
            response = requests.get('https://github.com/kstats/CIMA/archive/refs/heads/master/pictures.zip')
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('.cache/cima/images')

        for key in lines:
            item = lines[key]
            dialouge = []
            role = "tutor"
            for turn in item['past_convo']:
                dialouge.append({"role": role, "content": turn})
                role = "student" if role == "tutor" else "tutor"
            img_name = item['img'].split('/')[-1].replace('"', '')
            img_path = '.cache/cima/images/CIMA-master/pictures/' + img_name
            topic = img_name.split('.')[0]
            self.conversations.append(Conversation(problem = Problem(img = img_path), dialouge = dialouge, topic = topic))

class CoMTA(DialougeDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = 'https://raw.githubusercontent.com/Khan/tutoring-accuracy-dataset/refs/heads/main/CoMTA_dataset.json'
        self.conversations = []
        lines = json.loads(requests.get(path).text)
        for item in lines:
            dialouge = []
            topic = item['math_level']
            for turn in item['data']:
                if turn['role'] == 'assistant':
                    dialouge.append({"role": "tutor", "content": turn['content']})
                else:
                    dialouge.append({"role": "student", "content": turn['content']})
            if item['expected_result'] == 'Answer Accepted':
                dialouge.append({"role": "tutor", "content": "that is correct"})
            else:
                dialouge.append({"role": "tutor", "content": "that is incorrect"})
            self.conversations.append(Conversation(dialouge = dialouge, topic = topic))

def load_dataset(dataset_name):
    if dataset_name == 'stepwise_verify':
        return StepVerify()
    elif dataset_name == 'mathdial':
        return MathDial()
    elif dataset_name == 'bridge':
        return Bridge()
    elif dataset_name == 'cima':
        return Cima()
    elif dataset_name == 'comta':
        return CoMTA()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

# print('Stepwise Verify')
# dataset = load_dataset('stepwise_verify')
# print(dataset.conversations[0])

# print('MathDial')
# dataset = load_dataset('mathdial')
# print(dataset.conversations[0])

# print('Bridge')
# dataset = load_dataset('bridge')
# print(dataset.conversations[0])

# print('Cima')
# dataset = load_dataset('cima')
# print(dataset.conversations[0])

# print('CoMTA')
# dataset = load_dataset('comta')
# print(dataset.conversations[0])