
from typing import Dict, List
import re

try: 
    from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string,remove_boxed
    from lm_eval.tasks.gpqa.zeroshot.utils import process_docs

except ImportError as e:
    error_message = "Could not import library from 'lm_eval'. Please install the required package "
    raise ImportError(error_message)

# import dataset

def doc_to_target(doc):
    return postprocess_target(doc["answer"])

def postprocess_target(s):
    return str(s).strip()

def postprocess(output):

    response = re.sub(r".*?<\/think>(\\n)*", "", output, flags=re.DOTALL).strip()

    try:
        answer = remove_boxed(last_boxed_only_string(response))
        return answer.strip()
    except Exception:
        return output


def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return "".join(answers)

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    
    candidate = postprocess(results[0])
    gold = postprocess_target(doc["answer"])
    retval = 0

    if not gold:
        print(doc, candidate, gold)
    
    if is_equiv(candidate, gold):
        retval = 1

    results = {
        "exact_match": retval,
    }
    return results


def process_gpqa_docs(dataset):
    return process_docs(dataset)

def postprocess_target_gpqa(s):
    if s.startswith("(") and s.endswith(")"):
        return s[1:-1]
    return s

def process_results_gpqa(doc: dict, results: List[str]) -> Dict[str, int]:
    
    candidate = postprocess(results[0])
    gold = postprocess_target_gpqa(doc["answer"])
    retval = 0

    if not gold:
        print(doc, candidate, gold)
    
    if is_equiv(candidate, gold):
        retval = 1

    results = {
        "exact_match": retval,
        "response":candidate,
        "actual":gold

    }
    return results


def process_results_offline(response,answer):
    candidate = postprocess(response)
    gold = postprocess_target(answer)
    retval = 0
    
    if is_equiv(candidate, gold):
        retval = 1
    results = {
        "exact_match": retval,
        "response":candidate,
        "actual":gold
    }
    return results

