import json
import requests
from PIL import Image
import zipfile
import os
import hashlib
import datasets
from tqdm import tqdm

import re
from itertools import islice

from transformers import ImageGPTForCausalImageModeling

class Problem:
    def __init__(self, text = '', img = None):
        self.text = text
        if img is not None:
            
            if isinstance(img, str):
                if os.path.exists(img):
                    self.img = Image.open(img).convert('RGB')
                else:
                    self.img = None
            else:
                self.img = img
        else:
            self.img = None

    def __str__(self):
        return f"Problem: {self.text}\n"

class Conversation:
    def __init__(self, problem:Problem = None, solution = '', topic = '', dialogue = []):
        if problem is None:
            self.problem = Problem()
        else:
            self.problem = problem
        self.solution = solution
        self.topic = topic
        self.dialogue = [{"role": item['role'], "content": item['content']} for item in dialogue]
    
    def __str__(self):
        out = ""
        out += f"Question: {self.problem.text}\n"
        out += f"Solution: {self.solution}\n"
        out += f"Topic: {self.topic}\n"
        out += f"dialogue: \n"
        for turn in self.dialogue:
            out += f"{turn['role']}: {turn['content']}\n"
        out += f"--------------------------------\n"
        return out

def hash_question(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()

class Dataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

class GSM8K(Dataset):
    def __init__(self, split = "test"):
        self.dataset_name = "gsm8k"
        self.split = split
        self.subset = "main"
        self.hf_name = "openai/gsm8k"
        self.dataset = datasets.load_dataset(self.hf_name, self.subset)[self.split]

    
    def process_dataset(self):
        self.dataset = self.dataset.map(lambda x: {"question": x['question'], "answer": x['answer'].split("####")[1].strip()})
        self.dataset = self.dataset.map(lambda x: {"id": hash_question(x["question"])})
        return self.dataset
    
    
class DialogueDataset:
    def __init__(self, conversations = [], subset = 'train'):
        self.subset = subset
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]

    def get_dialogue(self, flip = False):
        conversations = []
        for conversation in self.conversations:
            dialogue = []
            for turn in conversation.dialogue:
                if flip:
                    dialogue.append({"role": 'assistant' if turn['role'] == 'student' else 'user', "content": turn['content']})
                else:
                    dialogue.append({"role": 'assistant' if turn['role'] == 'tutor' else 'user', "content": turn['content']})
            conversations.append(dialogue)
        return conversations

class StepVerify(DialogueDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = 'https://raw.githubusercontent.com/eth-lre/verify-then-generate/refs/heads/main/dataset/dataset.json'
        self.conversations = []
        for item in json.loads(requests.get(path).text):
            dialogue = []
            for turn in item['dialog_history']:
                role = "student" if turn['user'] == "Student" else "tutor"
                dialogue.append({"role": role , "content": turn['text']})
            dialogue.append({"role": "student" , "content": item['student_correct_response']})
            self.conversations.append(Conversation(problem = Problem(text = item['problem']), solution = item['reference_solution'], topic = item['topic'], dialogue = dialogue))

class MathDial(DialogueDataset):
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
            dialogue = []
            
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
                    dialogue.append({"role": "tutor" , "content": content})
                else:
                    dialogue.append({"role": "student" , "content": content})
            dialogue.append({"role": "student" , "content": item['student_incorrect_solution']})
            self.conversations.append(Conversation(problem = Problem(text = item['question']), solution = item['ground_truth'], topic = item['student_profile'], dialogue = dialogue))

class Bridge(DialogueDataset):
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
            dialogue = []
            for turn in item['c_h']:
                dialogue.append({"role": turn['user'], "content": turn['text']})
            if 'c_revision' in item:
                for turn in item['c_revision']:
                    dialogue.append({"role": turn['user'], "content": turn['text']})
            self.conversations.append(Conversation(topic = item['lesson_topic'], dialogue = dialogue))

class Cima(DialogueDataset):
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
            dialogue = []
            role = "tutor"
            for turn in item['past_convo']:
                dialogue.append({"role": role, "content": turn})
                role = "student" if role == "tutor" else "tutor"
            img_name = item['img'].split('/')[-1].replace('"', '')
            img_path = '.cache/cima/images/CIMA-master/pictures/' + img_name
            topic = img_name.split('.')[0]
            self.conversations.append(Conversation(problem = Problem(img = img_path), dialogue = dialogue, topic = topic))

class CoMTA(DialogueDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = 'https://raw.githubusercontent.com/Khan/tutoring-accuracy-dataset/refs/heads/main/CoMTA_dataset.json'
        self.conversations = []
        lines = json.loads(requests.get(path).text)
        for item in lines:
            dialogue = []
            topic = item['math_level']
            for turn in item['data']:
                if turn['role'] == 'assistant':
                    dialogue.append({"role": "tutor", "content": turn['content']})
                else:
                    dialogue.append({"role": "student", "content": turn['content']})
            if item['expected_result'] == 'Answer Accepted':
                dialogue.append({"role": "tutor", "content": "that is correct"})
            else:
                dialogue.append({"role": "tutor", "content": "that is incorrect"})
            self.conversations.append(Conversation(dialogue = dialogue, topic = topic))

class SocraTeach(DialogueDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        path = 'https://raw.githubusercontent.com/Ljyustc/SocraticLM/refs/heads/main/data/SocraTeach_multi.json'
        self.conversations = []
        data = json.loads(requests.get(path).text)
        for item in data:
            items = data[item]
            question = items['question']
            solution = items['answer']
            
            for key in items['dialogues']:
                dialogue = []
                for turn in items['dialogues'][key]:
                    if "system" in turn:
                        dialogue.append({"role": "tutor", "content": turn['system']})
                    if "user" in turn:
                        dialogue.append({"role": "student", "content": turn['user']})
                self.conversations.append(Conversation(problem = Problem(text = question), solution = solution, dialogue = dialogue))

class TutorChat(DialogueDataset):
    """
    TutorChat – 80 k synthetic teacher–student conversations ⟨hf: princeton‑nlp/TutorChat⟩
    """
    HF_NAME = "princeton-nlp/TutorChat"          
    ROLE_MAP = {"assistant": "tutor",
                "teacher": "tutor",
                "user": "student",
                "student": "student"}

    def __init__(self, subset: str = "train"):
        super().__init__(subset=subset)

        raw = datasets.load_dataset(self.HF_NAME,
                                    split=self.subset,
                                    streaming=True)
                      
        self.conversations: list[Conversation] = []
        for row in islice(raw, None):         
            dialogue = self._row_to_dialogue(row)
            topic = ": ".join(row.get("textbook_folder", "").split("/")[1:4]).strip(': ')
            
            self.conversations.append(
                Conversation(topic=topic, dialogue=dialogue))
            
    def _row_to_dialogue(self, row) -> list[dict]:
        conv = row["conversation"] if "conversation" in row else None
        if conv and isinstance(conv, list):          
            mode = row.get("mode", "")
            dialogue = []
            for i, turn in enumerate(conv):
                # In open‑book mode the first element is the chapter snippet – skip it
                if mode == "openbook" and i == 0:
                    continue
                role = "assistant" if i % 2 == 0 else "user"
                dialogue.append({"role": self.ROLE_MAP[role], "content": turn})
            return dialogue



class TutorEval(Dataset):
    """
    TutorEval – 834 expert‑written questions about long textbook
    chapters; supports open‑book / closed‑book flags. ⟨hf: princeton‑nlp/TutorEval⟩
    """
    HF_NAME = "princeton-nlp/TutorEval"    

    def __init__(self, split: str = "train"):
        super().__init__("tutoreval")
        self.split = split
        self.dataset = datasets.load_dataset(self.HF_NAME,
                                             split=self.split,
                                             streaming=True)
        
    def process_dataset(self):
        def _simplify(x):
            return {
                "id": hash_question(x["question"]),
                "chapter": x["chapter"],
                "question": x["question"],
                "key_points": x["key_points"],
                "closed_book": x["closed_book"],
                "answer_in_chapter": x["answer_in_chapter"],
                "misleading_question": x["misleading_question"],
                "difficulty": x["difficulty"],
                "domain": x["domain"],
                "path_to_chapter": x["path_to_chapter"],
            }

        self.dataset = self.dataset.map(_simplify)
        return self.dataset  
    

class Book2Dial(Dataset):

    def __init__(self, subset = 'train'):
        self.subset = subset
        base_path = "https://raw.githubusercontent.com/eth-lre/book2dial/refs/heads/main/generated_dataset"
        topics = ['business', 'math', 'science', 'social']
        paths = [f"{base_path}/{topic}_Persona_high_info.jsonl" for topic in topics]
        self.conversations = []
        for path in paths:
            response_text = requests.get(path).text.strip()
            for line in tqdm(response_text.split('\n')):
                item = json.loads(line)
                history = item['history']
                dialogue = []
                for i, turn in enumerate(history):
                    if i % 2 == 0:
                        dialogue.append({"role": "tutor", "content": turn})
                    else:
                        dialogue.append({"role": "student", "content": turn})
                self.conversations.append(Conversation(dialogue = dialogue))

class TutorBench(DialogueDataset):
    def __init__(self, subset = 'train'):
        self.subset = subset
        self.dataset = datasets.load_dataset("tutorbench/tutorbench", split = "train")
        self.conversations = []
        for item in self.dataset:
            dialogue = []
            dialogue.append({"role": "student", "content": item['PROMPT']})
            dialogue.append({"role": "tutor", "content": item['UC1_INITIAL_EXPLANATION']})
            dialogue.append({"role": "student", "content": item['FOLLOW_UP_PROMPT']})
            self.conversations.append(Conversation(dialogue = dialogue, topic = item['SUBJECT'], problem = Problem(text = item['PROMPT'], img = item['Image'])))
    

        
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
    elif dataset_name == 'tutor_chat':
        return TutorChat()
    elif dataset_name == 'tutor_eval':
        return TutorEval()
    elif dataset_name == 'gsm8k':
        return GSM8K()
    elif dataset_name == 'socra_teach':
        return SocraTeach()
    elif dataset_name == 'book2dial':
        return Book2Dial()
    elif dataset_name == 'tutor_bench':
        return TutorBench()
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

if __name__ == "__main__":
    print('Stepwise Verify')
    dataset = load_dataset('stepwise_verify')
    print(dataset.conversations[0])

    print('MathDial')
    dataset = load_dataset('mathdial')
    print(dataset.conversations[0])

    print('Bridge')
    dataset = load_dataset('bridge')
    print(dataset.conversations[0])

    print('Cima')
    dataset = load_dataset('cima')
    print(dataset.conversations[0])

    print('CoMTA')
    dataset = load_dataset('comta')
    print(dataset.conversations[0])

    print('TutorChat')
    dataset = load_dataset('tutor_chat')
    print(dataset.conversations[0])

    print('TutorEval')
    dataset = load_dataset('tutor_eval')
    dataset = dataset.process_dataset()
    for i, item in enumerate(dataset):
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Key Points: {item['key_points']}")
        print(f"Closed Book: {item['closed_book']}")
        print(f"Answer in Chapter: {item['answer_in_chapter']}")
        print(f"Misleading Question: {item['misleading_question']}")
        print(f"Difficulty: {item['difficulty']}")
        print(f"Domain: {item['domain']}")
        print(f"Path to Chapter: {item['path_to_chapter']}")
        print('--------------------------------')
        break

    print('SocraTeach')
    dataset = load_dataset('socra_teach')
    print(dataset.conversations[0])

    print('Book2Dial')
    dataset = load_dataset('book2dial')
    print(dataset.conversations[0])

    print('TutorBench')
    dataset = load_dataset('tutor_bench')
    print(dataset.conversations[0])
    