from llm_client import LLMClient
import utility
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import metrics
import time
import os
import json
import csv
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Model_zero_shot:
    def __init__(
        self,
        dataset_path,
        model,
        requestFilePath,
        pathToOverallResults,
        isCoT=False,
        isZeroShot=True,
        isResponseLarge=False,
    ):
        self.llm_client = LLMClient()
        self.prompts = utility.getPromptsWithoutCoT(isCoT)
        self.llmParameters = utility.getLLMRequestParameters()
        self.y_true = []
        self.y_pred = []
        self.y_pred_parsed = []
        self.char_count = 0
        self.df_zero_shot = utility.getDataFields()
        self.columns = [
            "Model",
            "Request",
            "Response",
            "TargetResponse",
            "#Newlines",
            "#Max_token",
            "Parsed_response",
        ]
        self.requests = pd.DataFrame(columns=self.columns)
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
        if isZeroShot == True:
            if isCoT == False:
                inputs = utility.createInputBatchZeroShotWithoutCoT(model)
            else:
                if isResponseLarge == True:
                    inputs = utility.createInputBatchZeroShotLargeResponse(model)
                else:
                    inputs = utility.createInputBatchZeroShotWithCoT(model)
        else:
            if model == utility._MODEL_ADA:
                examples = utility.getAdaExamples()
            if model == utility._MODEL_BABBAGE:
                examples = utility.getBabbageExamples()
            if model == utility._MODEL_CURIE:
                examples = utility.getCurieExamples()
            if model == utility._MODEL_DAVINCI:
                examples = utility.getDavinciExamples()
            if model == utility._MODEL_CHATGPT:
                examples = utility.getChatgptExamples()
            inputs = utility.createInputBatchFewShot(model, examples)

        for item in inputs:
            print(item)
            if isZeroShot == True:
                result = self.sendRequestZeroShot(
                    dataset_path,
                    model=item["model"],
                    promptTemplate=item["promptTemplate"],
                    attachPrefix=item["attachPrefix"],
                    attachSuffix=item["attachSuffix"],
                    attachIntermediate=item["attachIntermediate"],
                    pathToRequestsFile=requestFilePath,
                    maxToken=utility.max_token,
                    isResponseLarge=isResponseLarge,
                )
                print("result = ", result)
                pd.DataFrame.from_dict([result]).to_csv(pathToOverallResults)
            else:
                result1 = self.sendRequestZeroShot(
                    dataset_path,
                    model=item["model"],
                    promptTemplate=item["promptTemplate"],
                    attachPrefix=item["attachPrefix"],
                    attachSuffix=item["attachSuffix"],
                    attachIntermediate=item["attachIntermediate"],
                    pathToRequestsFile=requestFilePath,
                    maxToken=100,
                )
                result2 = self.sendRequestChainOfThoughts(
                    model, pathToRequestsFile=requestFilePath
                )
                pd.DataFrame.from_dict([result2]).to_csv(pathToOverallResults)


    def _getLLMResponse(
        self, prompts_collection, data, model, maxToken=utility.max_token
    ):
        results = []
        null_responses = 0
        self.requestsCount = -1
        self.requestsCount = -1
        for i in prompts_collection:
            request_data = {
                "prompt": i,
                "max_tokens": maxToken,
                "temperature": utility.temperature,
                "top_p": 0.5,
                "best_of": utility.n, #not applicable for chatgt
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
                                    "#Newlines": 0,
                                    "#Max_token": maxToken,
                                    "Parsed_response": response["error"],
                                }
                            ],
                            columns=self.columns,
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
                                "#Newlines": 0,
                                "#Max_token": maxToken,
                                "Parsed_response": response["error"],
                            },
                            ignore_index=True,
                        )
            else:
                print('response["choices"] = ', response["choices"])
                print('data["target_scores"] = ', data["target_scores"])
                for j in response["choices"]:
                    self.requestsCount += 1
                    print("self.requestsCount = ", self.requestsCount)
                    res = str(j["text"]).strip(" ").lower()
                    newLines = res.count("\n")
                    results.append(res)
                    print("self.y_pred_parsed count = ", len(self.y_pred_parsed))
                    y_pred_parsed = self._parse_response(res.strip("\n"))
                    self.y_pred_parsed.append(y_pred_parsed)
                    self.char_count += response["usage"]["total_tokens"]
                    if res == "":
                        null_responses += 1
                    if self.requests.empty:
                        self.requests = pd.DataFrame(
                            [
                                {
                                    "Model": model,
                                    "Request": i[j["index"]],
                                    "Response": str(res).strip("\n"),
                                    "TargetResponse": data["target_scores"][
                                        str(self.requestsCount)
                                    ].lower(),
                                    "#Newlines": newLines,
                                    "#Max_token": maxToken,
                                    "Parsed_response": y_pred_parsed,
                                }
                            ],
                            columns=self.columns,
                        )
                    else:
                        self.requests = self.requests.append(
                            {
                                "Model": model,
                                "Request": i[j["index"]],
                                "Response": str(res).strip("\n"),
                                "TargetResponse": data["target_scores"][
                                    str(self.requestsCount)
                                ].lower(),
                                "#Newlines": newLines,
                                "#Max_token": maxToken,
                                "Parsed_response": y_pred_parsed,
                            },
                            ignore_index=True,
                        )
                    self.y_pred.append(res.strip("\n"))
                    self.y_true.append(
                        data["target_scores"][str(self.requestsCount)].lower()
                    )
                    self.df_zero_shot["Null_responses"] = null_responses
                    self.data = data
        print("Requests count = ", self.requestsCount + 1)
        return data

    def _getLLMResponseLarge(self, prompts_collection, data, model, maxToken=100):
        results = []
        null_responses = 0
        self.requestsCount = -1
        print("prompts_collection len = ", len(prompts_collection))
        self.requestsCount = -1
        for i in prompts_collection:
            request_data = {
                "prompt": i,
                "max_tokens": maxToken,
                "temperature": utility.temperature,
                "top_p": 0.5,
                "best_of": utility.n,  # not applicable for chatgt
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
                                    "#Newlines": 0,
                                    "#Max_token": maxToken,
                                    "Parsed_response": response["error"],
                                }
                            ],
                            columns=self.columns,
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
                                "#Newlines": 0,
                                "#Max_token": maxToken,
                                "Parsed_response": response["error"],
                            },
                            ignore_index=True,
                        )
            else:
                print('response["choices"] = ', response["choices"])
                print('data["target_scores"] = ', data["target_scores"])
                for j in response["choices"]:
                    self.requestsCount += 1
                    print("self.requestsCount = ", self.requestsCount)
                    res = str(j["text"]).strip(" ").lower()
                    print("res = ", res)
                    newLines = res.count("\n")
                    results.append(res)
                    print("self.y_pred_parsed count = ", len(self.y_pred_parsed))
                    y_pred_parsed = self._sentiment_analysis(res.strip("\n"))
                    self.y_pred_parsed.append(y_pred_parsed)
                    self.char_count += response["usage"]["total_tokens"]
                    if res == "":
                        null_responses += 1
                    if self.requests.empty:
                        self.requests = pd.DataFrame(
                            [
                                {
                                    "Model": model,
                                    "Request": i[j["index"]],
                                    "Response": str(res).strip("\n"),
                                    "TargetResponse": data["target_scores"][
                                        str(self.requestsCount)
                                    ].lower(),
                                    "#Newlines": newLines,
                                    "#Max_token": maxToken,
                                    "Parsed_response": y_pred_parsed,
                                }
                            ],
                            columns=self.columns,
                        )
                    else:
                        self.requests = self.requests.append(
                            {
                                "Model": model,
                                "Request": i[j["index"]],
                                "Response": str(res).strip("\n"),
                                "TargetResponse": data["target_scores"][
                                    str(self.requestsCount)
                                ].lower(),
                                "#Newlines": newLines,
                                "#Max_token": maxToken,
                                "Parsed_response": y_pred_parsed,
                            },
                            ignore_index=True,
                        )
                    self.y_pred.append(res.strip("\n"))
                    self.y_true.append(
                        data["target_scores"][str(self.requestsCount)].lower()
                    )
                    self.df_zero_shot["Null_responses"] = null_responses
                    self.data = data
        # print("errors % = ", (self.errors / self.requestsCount) * 100)
        print("Requests count = ", self.requestsCount + 1)
        return data

    def _getLLMResponseLargeGPT4(self, prompts_collection, data, model, maxToken=100):
        results = []
        null_responses = 0
        self.requestsCount = -1
        print("prompts_collection len = ", len(prompts_collection))

        self.requestsCount = -1
        for i in prompts_collection:
            for prompt in i:
                request_data = {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": maxToken,
                    "temperature": utility.temperature,
                    "top_p": 0.5,
                    #                 "best_of": utility.n, #not applicable for chatgt
                    #                 "stream": utility.stream,
                    "stop": utility.stop,
                }
                print("Sending request")
                print("request length = ", len(i))
                time.sleep(utility._DELAY)
                self.time_delay += utility._DELAY
                response = self.llm_client.send_request(model, request_data)

                print("response = ", response)
                print("Response received")
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
                                        "#Newlines": 0,
                                        "#Max_token": maxToken,
                                        "Parsed_response": response["error"],
                                    }
                                ],
                                columns=self.columns,
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
                                    "#Newlines": 0,
                                    "#Max_token": maxToken,
                                    "Parsed_response": response["error"],
                                },
                                ignore_index=True,
                            )
                else:
                    print('response["choices"] = ', response["choices"])
                    print('data["target_scores"] = ', data["target_scores"])
                    for j in response["choices"]:
                        self.requestsCount += 1
                        print("self.requestsCount = ", self.requestsCount)
                        res = str(j["message"]["content"]).strip(" ").lower()
                        print("res = ", res)
                        newLines = res.count("\n")
                        results.append(res)
                        print("self.y_pred_parsed count = ", len(self.y_pred_parsed))
                        y_pred_parsed = self._sentiment_analysis(res.strip("\n"))
                        self.y_pred_parsed.append(y_pred_parsed)
                        self.char_count += response["usage"]["total_tokens"]
                        if res == "":
                            null_responses += 1
                        if self.requests.empty:
                            self.requests = pd.DataFrame(
                                [
                                    {
                                        "Model": model,
                                        "Request": prompt,
                                        "Response": str(res).strip("\n"),
                                        "TargetResponse": data["target_scores"][
                                            str(self.requestsCount)
                                        ].lower(),
                                        "#Newlines": newLines,
                                        "#Max_token": maxToken,
                                        "Parsed_response": y_pred_parsed,
                                    }
                                ],
                                columns=self.columns,
                            )
                        else:
                            self.requests = self.requests.append(
                                {
                                    "Model": model,
                                    "Request": prompt,
                                    "Response": str(res).strip("\n"),
                                    "TargetResponse": data["target_scores"][
                                        str(self.requestsCount)
                                    ].lower(),
                                    "#Newlines": newLines,
                                    "#Max_token": maxToken,
                                    "Parsed_response": y_pred_parsed,
                                },
                                ignore_index=True,
                            )
                        self.y_pred.append(res.strip("\n"))
                        self.y_true.append(
                            data["target_scores"][str(self.requestsCount)].lower()
                        )
                        self.df_zero_shot["Null_responses"] = null_responses
                        self.data = data
        # print("errors % = ", (self.errors / self.requestsCount) * 100)
        print("Requests count = ", self.requestsCount + 1)
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
        maxToken=utility.max_token,
        isResponseLarge=False,
    ):
        self.char_count = 0
        self.y_true = []
        self.y_pred = []
        self.start_cpu_time = time.process_time()
        self.df_zero_shot["Model"] = model
        self.df_zero_shot["Prefix"] = attachPrefix
        self.df_zero_shot["Intermediate"] = attachIntermediate
        self.df_zero_shot["Suffix"] = attachSuffix
        self.df_zero_shot["Max_token"] = maxToken
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
                prefix = self.prompts["prefix"]
            if attachIntermediate == True:
                intermediate = self.prompts["intermediate"]
            if attachSuffix == True:
                suffix = self.prompts["suffix"]
            #             prompt = (
            #                 promptTemplate.format(
            #                     premise=data["premise"][str(i)],
            #                     prefix=prefix,
            #                     intermediate=intermediate,
            #                     hypothesis=data["hypothesis"][str(i)],
            #                     suffix=suffix,
            #                 ).strip(" ")
            #                 + "A:"
            #             )
            prompt = promptTemplate.format(
                premise=data["premise"][str(i)].split(".")[0]
                if "Fact:" in data["premise"][str(i)]
                else "Fact: " + data["premise"][str(i)],
                prefix=prefix,
                intermediate=intermediate,
                hypothesis=data["hypothesis"][str(i)].split("Q:")[1].split("?")[0]
                if "Q:" in data["hypothesis"][str(i)]
                else data["hypothesis"][str(i)],
                suffix=suffix,
            ).strip(" ")
            print("prompt = ", prompt)
            # self.char_count += len(prompt)
            if len(prompts) < utility._TOTAL_REQUESTS:
                prompts.append(prompt)
            else:
                flag = True
                prompts_collection.append(prompts)
                prompts = []
                prompts.append(prompt)
        if flag == False:
            prompts_collection.append(prompts)
        if isResponseLarge == False:
            data = self._getLLMResponse(prompts_collection, data, model, maxToken)
        else:
            print("here in large response")
            if model == utility._MODEL_GPT4:
                data = self._getLLMResponseLargeGPT4(prompts_collection, data, model)
            else:
                data = self._getLLMResponseLarge(prompts_collection, data, model)
        self.end_cpu_time = time.process_time()
        cpu_time = self._evaluate_time(self.start_cpu_time, self.end_cpu_time)
        print("self.requestsCount = ", self.requestsCount)
        self.df_zero_shot["CPU_time"] = cpu_time / (self.requestsCount + 1)
        print("CPU time = ", cpu_time)
        print("model = ", model)
        self._evaluate_performance()
        # MAU = 50*10^6, DAU=70% of MAU
        if model == utility._MODEL_ADA:
            print("ADA here")
            #             self.df_zero_shot["Cost"] = (
            #                 50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            #             ) * utility._MODEL_ADA_PRICE  # dollars
            self.df_zero_shot["Cost"] = (self.char_count * utility._MODEL_ADA_PRICE) / (
                self.requestsCount + 1
            )  # dollars
            print("self.char_count = ", self.char_count)
        elif model == utility._MODEL_BABBAGE:
            print("babbage here")
            #             self.df_zero_shot["Cost"] = (
            #                 50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            #             ) * utility._MODEL_BABBAGE_PRICE
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_BABBAGE_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_CURIE:
            print("curie here")
            #             self.df_zero_shot["Cost"] = (
            #                 50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            #             ) * utility._MODEL_CURIE_PRICE
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_CURIE_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_DAVINCI:
            print("davinci here")
            #             self.df_zero_shot["Cost"] = (
            #                 50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            #             ) * utility._MODEL_DAVINCI_PRICE
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_DAVINCI_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_CHATGPT:
            print("chatgpt here")
            #             self.df_zero_shot["Cost"] = (
            #                 50 * pow(10, 6) * 0.70 * ((self.char_count * 0.75) + utility.max_token)
            #             ) * utility._MODEL_CHATGPT_PRICE
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_CHATGPT_PRICE
            ) / (self.requestsCount + 1)
        if pathToRequestsFile != "":
            with open(pathToRequestsFile, "a", encoding="utf-8") as file:
                file.write(self.requests.to_csv(header=False, index=False))

        return self.df_zero_shot

    def sendRequestChainOfThoughts(self, model, pathToRequestsFile=""):
        self.char_count = 0
        self.y_true = []
        self.y_pred = []
        self.start_cpu_time = time.process_time()
        prompts_collection = []
        prompts = []
        print("self.requests = ", self.requests)
        for i in range(0, len(self.requests)):
            flag = False
            print("self.requests.iloc[i] = ", self.requests.iloc[i]["Response"])
            prompt = (
                str(self.requests.iloc[i]["Request"].split("Q:")[0])
                + str(self.requests.iloc[i]["Response"].lower().strip(" "))
                + " Q:"
                + str(self.requests.iloc[i]["Request"]).split("Q:")[1].split("Let's")[0]
                + "Therefore, the answer is A:```yes or no```"
            )
            print("prompt = ", prompt)
            # self.char_count += len(prompt)
            if len(prompts) < utility._TOTAL_REQUESTS:
                prompts.append(prompt)
            else:
                flag = True
                prompts_collection.append(prompts)
                prompts = []
                prompts.append(prompt)
        if flag == False:
            prompts_collection.append(prompts)
        data = self._getLLMResponse(
            prompts_collection, self.data, self.requests.iloc[i]["Model"], 100
        )
        self.end_cpu_time = time.process_time()
        cpu_time = self._evaluate_time(self.start_cpu_time, self.end_cpu_time)
        self.df_zero_shot["CPU_time"] = cpu_time
        print("CPU time = ", cpu_time)
        self._evaluate_performance()
        # MAU = 50*10^6, DAU=70% of MAU
        if model == utility._MODEL_ADA:
            print("ADA here")
            self.df_zero_shot["Cost"] = (self.char_count * utility._MODEL_ADA_PRICE) / (
                self.requestsCount + 1
            )  # dollars
            print("self.char_count = ", self.char_count)
        elif model == utility._MODEL_BABBAGE:
            print("babbage here")
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_BABBAGE_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_CURIE:
            print("curie here")
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_CURIE_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_DAVINCI:
            print("davinci here")
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_DAVINCI_PRICE
            ) / (self.requestsCount + 1)
        elif model == utility._MODEL_CHATGPT:
            print("chatgpt here")
            self.df_zero_shot["Cost"] = (
                self.char_count * utility._MODEL_CHATGPT_PRICE
            ) / (self.requestsCount + 1)
        if pathToRequestsFile != "":
            with open(pathToRequestsFile, "a") as file:
                file.write(self.requests.to_csv(header=False, index=False))

        return self.df_zero_shot

    def _parse_response(self, y_pred_response):
        if "yes" in y_pred_response.lower():
            return "yes"
        elif "no" in y_pred_response.lower():
            return "no"
        else:
            return y_pred_response

    def _evaluate_performance(self):
        print("y_true = ", self.y_true)
        print("y_pred = ", self.y_pred_parsed)

        accuracy = accuracy_score(self.y_true, self.y_pred_parsed)
        self.df_zero_shot["Accuracy"] = accuracy
        metrics_report = metrics.classification_report(
            self.y_true, self.y_pred_parsed, digits=3, output_dict=True
        )
        self.df_zero_shot["Precision (wt. avg)"] = metrics_report["weighted avg"][
            "precision"
        ]
        self.df_zero_shot["Recall (wt. avg)"] = metrics_report["weighted avg"]["recall"]
        self.df_zero_shot["F1_score (wt. avg)"] = metrics_report["weighted avg"][
            "f1-score"
        ]
        print("total inputs = ", self.requestsCount + 1)
        print(accuracy)
        print(metrics_report)

    def _evaluate_time(self, start_time, end_time):
        return end_time - start_time

    def _sentiment_analysis(self, response):
        print("response in sentiment analysis = ", response)
        sid_obj = SentimentIntensityAnalyzer()
        sentiment_dict = sid_obj.polarity_scores(response)
        if sentiment_dict["compound"] >= 0.05:
            return "yes"
        elif sentiment_dict["compound"] <= -0.05:
            return "no"
        else:
            return "no"
