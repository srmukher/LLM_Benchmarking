import json
import pandas as pd

llm_base_path = ""
llm_api_version = ""
api_key = ""
api_key_gpt4 = ""
azure_resource_group = ""
max_token = 3
temperature = 1
n = 1
stream = False
stop = "None"
_MODEL_ADA = "text-ada-001"
_MODEL_BABBAGE = "text-babbage-001"
_MODEL_CURIE = "text-curie-001"
_MODEL_DAVINCI = "text-davinci-003"
_MODEL_CHATGPT = "gpt35turbo"
_MODEL_GPT4 = "GPT4"
_MODEL_ADA_PRICE = 0.0004 / 1000
_MODEL_BABBAGE_PRICE = 0.0005 / 1000
_MODEL_CURIE_PRICE = 0.0020 / 1000
_MODEL_DAVINCI_PRICE = 0.020 / 1000
_MODEL_CHATGPT_PRICE = 0.002 / 1000
_TOTAL_REQUESTS = 10
_DELAY = 15
_ENTAILED_POLARITY_DATASET = "EntailedPolarity"
_ANALYTICAL_ENTAILMENT_DATASET = "AnalyticalEntailment"
_SUPERGLUE_RTE_DATASET = "SuperGLUE_RTE"

def getPromptsWithoutCoT(isCoT):
    if isCoT == True:
        prefix = "Given the fact, answer the following question with **yes/no** and provide the steps used to get the answer."
        suffix = "Given the previous fact, answer the following question with **yes/no** and provide the steps used to get the answer."
        intermediate = "Given the previous fact, answer the following question with **yes/no** and provide the steps used to get the answer."
    else:
        prefix = "Given the fact, answer the following question with **yes/no**."
        suffix = "Given the previous fact, answer the following question with **yes/no**."
        intermediate = "Given the previous fact, answer the following question with **yes/no**."
    return {'prefix': prefix, 'suffix': suffix, 'intermediate': intermediate}

def createInputBatchZeroShotWithCoT(model):
    return [
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? Let's think step by step. Answer in **yes/no** only. A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{prefix} {premise}. Q:{hypothesis}? Let's think step by step. Answer in **yes/no** only. A:",
            "attachPrefix": True,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {intermediate} Q:{hypothesis}? Let's think step by step. Answer in **yes/no** only. A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": True,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? {suffix} Let's think step by step. Answer in **yes/no**. A:",
            "attachPrefix": False,
            "attachSuffix": True,
            "attachIntermediate": False,
        },
    ]

def createInputBatchZeroShotLargeResponse(model):
    return [
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? Let's think step by step. A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{prefix} {premise}. Q:{hypothesis}? Let's think step by step. A:",
            "attachPrefix": True,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {intermediate} Q:{hypothesis}? Let's think step by step. A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": True,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? {suffix} Let's think step by step. A:",
            "attachPrefix": False,
            "attachSuffix": True,
            "attachIntermediate": False,
        },
    ]


def createInputBatchZeroShotWithoutCoT(model):
    return [
         {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? yes/no? A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{prefix}. {premise}. Q:{hypothesis}? A:",
            "attachPrefix": True,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {intermediate} Q:{hypothesis}? A:",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": True,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? {suffix}. A:",
            "attachPrefix": False,
            "attachSuffix": True,
            "attachIntermediate": False,
        },
    ]

def createInputBatchFewShot(model, example):
    return [
        {
            "model": model,
            "promptTemplate": example + "\n" + "{premise}. Q:{hypothesis}? Let's think step by step. Therefore, the answer is A:```yes or no```",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{prefix}. {premise}. Q:{hypothesis}? Let's think step by step. A: ```yes or no```",
            "attachPrefix": True,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {intermediate}. Q:{hypothesis}? Let's think step by step. A: ```yes or no```",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": True,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. Q:{hypothesis}? {suffix}. Let's think step by step. A: ```yes or no```",
            "attachPrefix": False,
            "attachSuffix": True,
            "attachIntermediate": False,
        },
    ]


def getLLMRequestParameters():
    return {"temperature": 0, "max_tokens": 1, "n": 1, "stream": False, "stop": ["\n"]}


def getDataFields():
    return {
        "Model": "",
        "Max_token": "",
        "Prefix": "",
        "Intermediate": "",
        "Suffix": "",
        "CPU_time": "",
        "Accuracy": 0.00,
        "Precision (wt. avg)": 0.00,
        "Recall (wt. avg)": 0.00,
        "F1_score (wt. avg)": 0.00,
        "Null_responses": 0,
        "Cost": 0.00,
    }


def formatEntailedPolarity(original_dataset_path):
    f = open(original_dataset_path, encoding="utf8")
    data = json.load(f)
    formattedData = pd.DataFrame()
    for item in data["examples"]:
        sentence = item["input"].split(".")
        if formattedData.empty:
            formattedData = pd.DataFrame(
                [
                    {
                        "input": item["input"],
                        "premise": sentence[0],
                        "hypothesis": sentence[1],
                        "target_scores": "yes"
                        if item["target_scores"]["yes"] == 1.0
                        else "no",
                    }
                ],
                columns=["input", "premise", "hypothesis", "target_scores"],
            )
        else:
            print(item["input"])
            formattedData = formattedData.append(
                {
                    "input": item["input"],
                    "premise": sentence[0],
                    "hypothesis": sentence[1],
                    "target_scores": "yes"
                    if item["target_scores"]["yes"] == 1.0
                    else "no",
                },
                ignore_index=True,
            )
    formattedData.to_csv("../Dataset/Preprocessed/EntailedPolarityFormatted.csv")
    formattedData.to_json("../Dataset/Preprocessed/EntailedPolarityFormatted.json")
    return formattedData


def formatAnalyticalEntailment(original_dataset_path):
    f = open(original_dataset_path, encoding="utf8")
    data = json.load(f)
    formattedData = pd.DataFrame()
    for item in data["examples"]:
        sentence = item["input"].split(".")
        if formattedData.empty:
            formattedData = pd.DataFrame(
                [
                    {
                        "input": item["input"],
                        "premise": sentence[0],
                        "hypothesis": sentence[1],
                        "target_scores": "yes"
                        if item["target_scores"]["entailment"] == 1.0
                        else "no",
                    }
                ],
                columns=["input", "premise", "hypothesis", "target_scores"],
            )
        else:
            formattedData = formattedData.append(
                {
                    "input": item["input"],
                    "premise": sentence[0],
                    "hypothesis": sentence[1],
                    "target_scores": "yes"
                    if item["target_scores"]["entailment"] == 1.0
                    else "no",
                },
                ignore_index=True,
            )
    formattedData.to_csv("../Dataset/Preprocessed/AnalyticalEntailmentFormatted.csv")
    formattedData.to_json("../Dataset/Preprocessed/AnalyticalEntailmentFormatted.json")
    return formattedData


def formatSuperGlueRTE(original_dataset_path):
    with open(original_dataset_path, "r") as json_file:
        json_list = list(json_file)
    formattedData = pd.DataFrame()
    for json_str in json_list:
        item = json.loads(json_str)
        if formattedData.empty:
            formattedData = pd.DataFrame(
                [
                    {
                        "input": item["premise"] + " " + item["hypothesis"],
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "target_scores": "yes"
                        if item["label"] == "entailment"
                        else "no",
                    }
                ],
                columns=["input", "premise", "hypothesis", "target_scores"],
            )
        else:
            formattedData = formattedData.append(
                {
                    "input": item["premise"] + " " + item["hypothesis"],
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "target_scores": "yes" if item["label"] == "entailment" else "no",
                },
                ignore_index=True,
            )
    formattedData.to_csv("../Dataset/Preprocessed/SuperGLUEFormatted.csv")
    formattedData.to_json("../Dataset/Preprocessed/SuperGLUEFormatted.json")
    return formattedData


def formatData():
    formatEntailedPolarity("../Dataset/Original/Entailed_Polarity_task.json")
    formatAnalyticalEntailment("../Dataset/Original/Analytical_entailment_task.json")
    formatSuperGlueRTE("../Dataset/Original/SuperGLUE_train_task.jsonl")

def getAdaExamples():
    return "Example: 1. The meeting starts in less than an hour. yes Q: So the meeting starts in less than ten minutes? Therefore, the answer is A:```yes or no```, A: Yes, 2. Lina met two nurses. yes, lina met two nurses. Q: So, Lina met at least one woman? Therefore, the answer is A:```yes or no```, A: No "

def getBabbageExamples():
    return "1. The meeting starts in less than an hour. Q: So the meeting starts in less than ten minutes? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes, 2. Lina met two nurses. Q: So, Lina met at least one woman? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes "
    
def getCurieExamples():
    return "1. The meeting starts in less than an hour. Q: So the meeting starts in less than ten minutes? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes, 2. Lina met two nurses. Q: So, Lina met at least one woman? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes "

def getDavinciExamples():
    return "1. The meeting starts in less than an hour. Q: So the meeting starts in less than ten minutes? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes, 2. Lina met two nurses. Q: So, Lina met at least one woman? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes "

def getChatgptExamples():
    return "1. The meeting starts in less than an hour. Q: So the meeting starts in less than ten minutes? Let's think step by step. Therefore, the answer is A:```yes or no```, A: Yes, 2. Lina met two nurses. Q: So, Lina met at least one woman? Let's think step by step. Therefore, the answer is A:```yes or no```, A: No "