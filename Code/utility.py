import json
import pandas as pd

llm_base_path = ""
llm_api_version = ""
api_key = ""
azure_resource_group = ""
prefix = "Given the fact, answer the following question with a yes or no."
suffix = "Given the previous fact, answer the following question with a yes or a no."
intermediate = (
    "Given the previous fact, answer the following question with a yes or a no."
)
max_token = 1
temperature = 0
n = 1
stream = False
stop = ["\n"]
_MODEL_ADA = "text-ada-001"
_MODEL_BABBAGE = "text-babbage-001"
_MODEL_CURIE = "text-curie-001"
_MODEL_DAVINCI = "text-davinci-003"
_MODEL_CHATGPT = "gpt35turbo"
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


def getPrompts():
    return {
        "Prompt1": "{premise}. {hypothesis}?",  # no instruction
        "Prompt2": "{premise}. {hypothesis} yes or no?",  # no instruction  with response options (Yes/No)
        "Prompt3": "{prefix}. {premise}. {hypothesis}.",  # instruction with only prefix
        "Prompt4": "{premise}. {intermediate}. {hypothesis}.",  # instruction with only intermediate instruction
        "Prompt5": "{premise}. {hypothesis}? {suffix}.",  # instruction with only suffix
    }


def createInputBatch(model):
    return [
        {
            "model": model,
            "promptTemplate": "{premise}. {hypothesis}?",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {hypothesis} yes or no?",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{prefix}. {premise}. {hypothesis}.",
            "attachPrefix": True,
            "attachSuffix": False,
            "attachIntermediate": False,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {intermediate}. {hypothesis}.",
            "attachPrefix": False,
            "attachSuffix": False,
            "attachIntermediate": True,
        },
        {
            "model": model,
            "promptTemplate": "{premise}. {hypothesis}? {suffix}.",
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
