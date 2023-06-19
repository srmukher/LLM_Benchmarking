from llm_client import LLMClient
import utility
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
import os
import json
import csv


class Model_zero_shot:
    def __init__(self, dataset_path, model):
        self.llm_client = LLMClient()
        self.prompts = utility.getPrompts()
        self.llmParameters = utility.getLLMRequestParameters()
        self.y_true = []
        self.y_pred = []
        self.char_count = 0
        self.df_zero_shot = utility.getDataFields()
        self.requests = pd.DataFrame(
            columns=["Model", "Request", "Response", "TargetResponse"]
        )
        self.data = []
        self.errors = 0
        self.requestsCount = 0
        self.time_delay = 0
        self.start_wall_time = 0
        self.start_cpu_time = 0
        self.end_wall_time = 0
        self.end_cpu_time = 0
        if (
            not os.path.exists("../Dataset/Preprocessed/EntailedPolarityFormatted.csv")
            or not os.path.exists(
                "../Dataset/Preprocessed/AnalyticalEntailmentFormatted.csv"
            )
            or not os.path.exists("../Dataset/Preprocessed/SuperGLUEFormatted.csv")
            or not os.path.exists(
                "../Dataset/Preprocessed/EntailedPolarityFormatted.json"
            )
            or not os.path.exists(
                "../Dataset/Preprocessed/AnalyticalEntailmentFormatted.json"
            )
            or not os.path.exists("../Dataset/Preprocessed/SuperGLUEFormatted.json")
        ):
            utility.formatData()
        inputs = utility.createInputBatch(model)

        for item in inputs:
            print(item)
            result = self.sendRequestZeroShot(
                dataset_path,
                model=item["model"],
                promptTemplate=item["promptTemplate"],
                attachPrefix=item["attachPrefix"],
                attachSuffix=item["attachSuffix"],
                attachIntermediate=item["attachIntermediate"],
            )
            print("result = ", result)
            print(df_zero_shot)
            df_zero_shot = pd.concat(
                [df_zero_shot, pd.DataFrame.from_dict([result])], ignore_index=True
            )
        return df_zero_shot

    def _getLLMResponse(self, prompts_collection, data, model):
        results = []
        null_responses = 0
        count = 0
        print("prompts_collection len = ", len(prompts_collection))
        for i in prompts_collection:
            count = 0
            request_data = {
                "prompt": i,
                "max_tokens": utility.max_token,
                "temperature": utility.temperature,
                "n": utility.n,
                "stream": utility.stream,
                "stop": utility.stop,
            }
            print("Sending request")
            print("request length = ", len(i))
            time.sleep(utility._DELAY)
            self.time_delay += utility._DELAY
            response = self.llm_client.send_request(model, request_data)

            print("response = ", response)
            print("Response received")
            print("current request count = ", count)
            print("response count = ", len(response))
            if "error" in response:
                print("Error: ", response["error"])
                self.errors += 1
                for errorSet in range(len(i)):
                    results.append(response["error"])
                    if self.requests.empty:
                        self.requests = pd.DataFrame(
                            [
                                {
                                    "Model": model,
                                    "Request": i[errorSet],
                                    "Response": response["error"],
                                    "TargetResponse": data["target_scores"][
                                        str(errorSet)
                                    ].lower(),
                                }
                            ],
                            columns=["Model", "Request", "Response", "TargetResponse"],
                        )
                    else:
                        self.requests = self.requests.append(
                            {
                                "Model": model,
                                "Request": i[errorSet],
                                "Response": response["error"],
                                "TargetResponse": data["target_scores"][
                                    str(errorSet)
                                ].lower(),
                            },
                            ignore_index=True,
                        )
                    self.requestsCount += 1
            else:
                for j in response["choices"]:
                    print("self.requestsCount = ", self.requestsCount)
                    res = str(j["text"]).strip(" ").lower()
                    results.append(res)
                    if res == "":
                        null_responses += 1
                    self.y_pred.append(res)
                    self.y_true.append(
                        data["target_scores"][str(self.requestsCount)].lower()
                    )
                    if self.requests.empty:
                        self.requests = pd.DataFrame(
                            [
                                {
                                    "Model": model,
                                    "Request": i[count],
                                    "Response": res,
                                    "TargetResponse": data["target_scores"][
                                        str(count)
                                    ].lower(),
                                }
                            ],
                            columns=["Model", "Request", "Response", "TargetResponse"],
                        )
                    else:
                        self.requests = self.requests.append(
                            {
                                "Model": model,
                                "Request": i[count],
                                "Response": res,
                                "TargetResponse": data["target_scores"][
                                    str(count)
                                ].lower(),
                            },
                            ignore_index=True,
                        )
                    self.df_zero_shot["Null_responses"] = null_responses
                    self.data = data
                    count = count + 1
                    self.requestsCount += 1
        print("errors % = ", (self.errors / self.requestsCount) * 100)
        print("Requests count = ", self.requestsCount)
        return data

    def sendRequestZeroShot(
        self,
        dataset_path,
        model,
        promptTemplate,
        attachPrefix=False,
        attachSuffix=False,
        attachIntermediate=False,
        pathToRequestsFile="",
    ):
        self.char_count = 0
        self.y_true = []
        self.y_pred = []
        self.start_cpu_time = time.process_time()
        self.df_zero_shot["Model"] = model
        self.df_zero_shot["Prefix"] = attachPrefix
        self.df_zero_shot["Intermediate"] = attachIntermediate
        self.df_zero_shot["Suffix"] = attachSuffix
        self.df_zero_shot["Max_token"] = utility.max_token
        f = open(dataset_path, encoding="utf8")
        data = json.load(f)
        f.close()
        prompts_collection = []
        prompts = []
        for i in range(0, len(data["input"])):
            flag = False
            print("premise = ", data["premise"][str(i)])
            prefix = ""
            intermediate = ""
            suffix = ""
            if attachPrefix == True:
                prefix = utility.prefix
            if attachIntermediate == True:
                intermediate = utility.intermediate
            if attachSuffix == True:
                suffix = utility.suffix
            prompt = (
                promptTemplate.format(
                    premise=data["premise"][str(i)],
                    prefix=prefix,
                    intermediate=intermediate,
                    hypothesis=data["hypothesis"][str(i)],
                    suffix=suffix,
                ).strip(" ")
                + "A:"
            )
            print("prompt = ", prompt)
            self.char_count += len(prompt)
            if len(prompts) < utility._TOTAL_REQUESTS:
                prompts.append(prompt)
            else:
                flag = True
                prompts_collection.append(prompts)
                prompts = []
                prompts.append(prompt)
        if flag == False:
            prompts_collection.append(prompts)
        data = self._getLLMResponse(prompts_collection, data, model)
        self.end_cpu_time = time.process_time()
        cpu_time = self._evaluate_time(self.start_cpu_time, self.end_cpu_time)
        self.df_zero_shot["CPU_time"] = cpu_time
        print("CPU time = ", cpu_time)
        print("model = ", model)
        self._evaluate_performance()
        # MAU = 50*10^6, DAU=70% of MAU
        if model == utility._MODEL_ADA:
            print("ADA here")
            self.df_zero_shot["Cost"] = (
                50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            ) * utility._MODEL_ADA_PRICE  # dollars
            print("self.char_count = ", self.char_count)
        elif model == utility._MODEL_BABBAGE:
            print("babbage here")
            self.df_zero_shot["Cost"] = (
                50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            ) * utility._MODEL_BABBAGE_PRICE
        elif model == utility._MODEL_CURIE:
            print("curie here")
            self.df_zero_shot["Cost"] = (
                50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            ) * utility._MODEL_CURIE_PRICE
        elif model == utility._MODEL_DAVINCI:
            print("davinci here")
            self.df_zero_shot["Cost"] = (
                50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            ) * utility._MODEL_DAVINCI_PRICE
        elif model == utility._MODEL_CHATGPT:
            print("chatgpt here")
            self.df_zero_shot["Cost"] = (
                50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            ) * utility._MODEL_CHATGPT_PRICE
        if pathToRequestsFile != "":
            with open(
                "../Dataset/Response/Analytical_Entailment_requests_Ada.csv", "a"
            ) as file:
                file.write(self.requests.to_csv(header=False, index=False))

        return self.df_zero_shot

    def _evaluate_performance(self):
        print("y_true = ", self.y_true)
        print("y_pred = ", self.y_pred)
        accuracy = accuracy_score(self.y_true, self.y_pred)
        self.df_zero_shot["Accuracy"] = accuracy
        metrics_report = metrics.classification_report(
            self.y_true, self.y_pred, digits=3, output_dict=True
        )
        self.df_zero_shot["Precision (wt. avg)"] = metrics_report["weighted avg"][
            "precision"
        ]
        self.df_zero_shot["Recall (wt. avg)"] = metrics_report["weighted avg"]["recall"]
        self.df_zero_shot["F1_score (wt. avg)"] = metrics_report["weighted avg"][
            "f1-score"
        ]
        print("total inputs = ", self.requestsCount)
        print(accuracy)
        print(metrics_report)

    def _evaluate_time(self, start_time, end_time):
        return end_time - start_time
