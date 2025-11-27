import json
import requests
from PIL import Image
import zipfile
import os
import hashlib
import datasets

from torch.utils.data import Dataset
import re
from itertools import islice

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




# -------------------------------------------- VLMs Datasets --------------------------------------------
class MathVision(Dataset):
    def __init__(self, split="test"):
        self.dataset_name = "MathVision"
        self.split = split
        self.hf_name = "MathLLMs/MathVision"
        self.dataset = datasets.load_dataset(self.hf_name)[self.split]

    def _normalize(self, x):
        return {
            "id": x.get("id", hash_question(x.get("question", ""))),
            "question": x.get("question", ""),
            "option": x.get("option", []),
            "image_path": x.get("image", None),
            "image": x.get("decoded_image", None),
            "answer": x.get("answer", None),
            "solution": x.get("solution", None),
            "level": x.get("level", None),
            "subject": x.get("subject", None),
        }

    def process_dataset(self):
        self.dataset = self.dataset.map(lambda x: self._normalize(x))
        return self.dataset

class ScienceQA(Dataset):
    def __init__(self, split="test"):
        self.dataset_name = "ScienceQA"
        self.split = split
        self.hf_name = "derek-thomas/ScienceQA"
        self.dataset = datasets.load_dataset(self.hf_name)[self.split]

    def _normalize_example(self, x):
        image = x.get("image") or x.get("img") or None

        question = x.get("question") or x.get("prompt") or ""

        choices = x.get("choices") or x.get("options") or []
        if isinstance(choices, str):
            try:
                import ast
                choices = ast.literal_eval(choices)
            except Exception:
                choices = [choices]

        raw_ans = x.get("answer", None)
        answer = None
        if raw_ans is None:
            answer = None
        else:
            if isinstance(raw_ans, int):
                answer = raw_ans
            else:
                try:
                    answer = int(raw_ans)
                except Exception:
                    try:
                        answer = choices.index(raw_ans)
                    except Exception:
                        answer = None

        return {
            "image": image,
            "question": question,
            "choices": choices,
            "answer": answer,
            "hint": x.get("hint", None),
            "task": x.get("task", None),
            "grade": x.get("grade", None),
            "subject": x.get("subject", None),
            "topic": x.get("topic", None),
            "category": x.get("category", None),
            "skill": x.get("skill", None),
            "lecture": x.get("lecture", None),
            "solution": x.get("solution", None),
        }

    def process_dataset(self):
        self.dataset = self.dataset.map(lambda x: self._normalize_example(x))
        self.dataset = self.dataset.map(lambda x: {"id": hash_question(x["question"])})
        return self.dataset


# -------------------------------------------- Bio/Medical Datasets --------------------------------------------
class BioASQ(Dataset):
    def __init__(self, split="test"):
        self.dataset_name = "BioASQ"
        self.split = split
        self.hf_name = "lucadiliello/bioasqqa"
        self.dataset = datasets.load_dataset(self.hf_name)[self.split]

    def _normalize(self, x):
        return {
            "id": x.get("key", None),
            "context": x.get("context", ""),
            "question": x.get("question", ""),
            "answers": x.get("answers", []),
            "labels": x.get("labels", []),
        }

    def process_dataset(self):
        self.dataset = self.dataset.map(lambda x: self._normalize(x))
        return self.dataset

class MoleculeQA(Dataset):
    def __init__(self, split="test"):
        self.hf_name = "hcaoaf/MoleculeQA"
        self.split = split
        self.dataset = datasets.load_dataset(self.hf_name)[self.split]

    def _normalize_example(self, rows_chunk):
        """
        Convert a 8-row chunk of text into a QA example.
        rows_chunk: list of 8 strings [prompt, SMILES, question, option A-D, answer]
        """
        try:
            cid = rows_chunk[1].split(": ", 1)[1].strip()
            question = rows_chunk[2].split(": ", 1)[1].strip()
            options = [r.split(": ", 1)[1].strip() for r in rows_chunk[3:7]]
            answer_letter = rows_chunk[7].strip().strip('"').strip("\t")
            key = ord(answer_letter.upper()) - ord("A")
            labels = [1 if j == key else 0 for j in range(4)]
            return {
                "id": hash_question(question),
                "context": cid,
                "question": question,
                "options": options,
                "answer": key,
                "labels": labels
            }
        except Exception:
            return None  

    def process_dataset(self):
        
        rows = [x['text'] for x in self.dataset]
        rows = rows[1:]

        examples = []
        i = 0
        while i + 7 < len(rows):
            chunk = rows[i:i+8]
            ex = self._normalize_example(chunk)
            if ex is not None:
                examples.append(ex)
            i += 8  

        self.dataset = examples
        return self.dataset

class PubMedQA:
    def __init__(self, path_to_json):
        """
        path_to_json: path to the pre-split JSON file, e.g. pqaa_dev_set.json
        """
        with open(path_to_json, "r") as f:
            self.dataset = json.load(f)

    def _normalize_example(self, info):
        """
        Normalize each entry to standard QA fields
        """
        question = info.get("QUESTION")
        context = info.get("CONTEXTS")           # abstract without conclusion
        long_answer = info.get("LONG_ANSWER")   # conclusion
        final_decision = info.get("final_decision")  # yes / no / maybe

        label_map = {"yes": 0, "no": 1, "maybe": 2}
        label = label_map.get(short_answer.lower(), None)

        return {
            "id": hash_question(question),
            "context": context,
            "question": question,
            "answer": long_answer,
            "final_decision": final_decision,
            "label": label
        }

    def process_dataset(self):
        """
        Convert the JSON dict into a list of normalized examples
        """
        processed = [self._normalize_example(info) for pmid, info in self.dataset.items()]
        self.dataset = processed
        return self.dataset

# -------------------------------------------- LLMs Datasets QA --------------------------------------------
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


# -------------------------------------------- Dialogue Datasets --------------------------------------------
class DialougeDataset:
    def __init__(self, conversations = [], subset = 'train'):
        self.subset = subset
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        return self.conversations[idx]

    def get_dialouge(self, flip = False):
        conversations = []
        for conversation in self.conversations:
            dialouge = []
            for turn in conversation.dialouge:
                if flip:
                    dialouge.append({"role": 'assistant' if turn['role'] == 'student' else 'user', "content": turn['content']})
                else:
                    dialouge.append({"role": 'assistant' if turn['role'] == 'tutor' else 'user', "content": turn['content']})
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

class SocraTeach(DialougeDataset):
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
                dialouge = []
                for turn in items['dialogues'][key]:
                    if "system" in turn:
                        dialouge.append({"role": "tutor", "content": turn['system']})
                    if "user" in turn:
                        dialouge.append({"role": "student", "content": turn['user']})
                self.conversations.append(Conversation(problem = Problem(text = question), solution = solution, dialouge = dialouge))

class TutorChat(DialougeDataset):
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
                Conversation(topic=topic, dialouge=dialogue))
            
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
    
    
def load_dataset(dataset_name, path_to_json=None):
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
    elif dataset_name == 'mathvision':
        return MathVision()
    elif dataset_name == 'scienceqa':
        return ScienceQA()
    elif dataset_name == 'bioasq':
        return BioASQ()
    elif dataset_name == 'moleculeqa':
        return MoleculeQA()
    elif dataset_name == 'socra_teach':
        return SocraTeach()
    elif dataset_name == 'pubmedqa':
        if path_to_json is None:
            raise ValueError("path_to_json must be provided for PubMedQA dataset")
        return PubMedQA(path_to_json)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

if __name__ == "__main__":
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

    # print('TutorChat')
    # dataset = load_dataset('tutor_chat')
    # print(dataset.conversations[0])

    # print('TutorEval')
    # dataset = load_dataset('tutor_eval')
    # dataset = dataset.process_dataset()

    # print('SocraTeach')
    # dataset = load_dataset('socra_teach')
    # print(dataset.conversations[0])

    # VLMs Dataset sample
    print('MathVision')
    dataset = load_dataset('mathvision')
    dataset = dataset.process_dataset()
    print(dataset[0])


    print('ScienceQA')
    dataset = load_dataset('scienceqa')
    dataset = dataset.process_dataset()
    print(dataset[0])
    
    # Bio/Medical Dataset 

    print('BioASQ')
    dataset = load_dataset('bioasq')
    dataset = dataset.process_dataset()
    print(dataset[0])

    print('MoleculeQA Test')
    dataset = load_dataset('moleculeqa')
    dataset = dataset.process_dataset()
    print(dataset[0])

    print('PubMedQA Test')
    dataset = load_dataset('pubmedqa', path_to_json='pqaa_dev_set.json')
    dataset = dataset.process_dataset()

    for i, item in enumerate(dataset):
        print(f"ID: {item['id']}")
        print(f"Question: {item['question']}")
        for label, key in [
            ("Key Points", "key_points"),
            ("Closed Book", "closed_book"),
            ("Answer in Chapter", "answer_in_chapter"),
            ("Misleading Question", "misleading_question"),
            ("Difficulty", "difficulty"),
            ("Domain", "domain"),
            ("Path to Chapter", "path_to_chapter"),
            ("Context", "context"),
            ("Answer", "answer"),
            ("Key", "key"),
            ("Image", "image"),
            ("Options", "options")
        ]:
            if key in item:
                print(f"{label}: {item[key]}")

        print('--------------------------------')
        break

   
    