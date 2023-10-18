import json
import os
import atexit
import requests
import utility


class LLMClient:
    _ENDPOINT = "{api_base_path}/{model}/completions?api-version={api_version}"
    _ENDPOINT_ADA = _ENDPOINT.format(
            api_base_path=utility.llm_base_path,
            model=utility._MODEL_ADA,
            api_version=utility.llm_api_version,
    )

    _ENDPOINT_BABBAGE = (
        _ENDPOINT.format(
            api_base_path=utility.llm_base_path,
            model=utility._MODEL_BABBAGE,
            api_version=utility.llm_api_version,
        ),
    )

    _ENDPOINT_CURIE = (
        _ENDPOINT.format(
            api_base_path=utility.llm_base_path,
            model=utility._MODEL_CURIE,
            api_version=utility.llm_api_version,
        ),
    )

    _ENDPOINT_DAVINCI = (
        _ENDPOINT.format(
            api_base_path=utility.llm_base_path,
            model=utility._MODEL_DAVINCI,
            api_version=utility.llm_api_version,
        ),
    )

    _ENDPOINT_CHATGPT = (
        _ENDPOINT.format(
            api_base_path=utility.llm_base_path,
            model=utility._MODEL_CHATGPT,
            api_version=utility.llm_api_version,
        ),
    )

    def send_request(self, model_name, request):
        if model_name == utility._MODEL_GPT4:
            key = utility.api_key_gpt4
        else:
            key = utility.api_key
        # populate the headers
        headers = {
            "authority": "{res_grp}.openai.azure.com".format(
                res_grp=utility.azure_resource_group
            ),
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "api-key": key,
            "content-type": "application/json",
            "origin": "https://oai.azure.com",
            "referer": "https://oai.azure.com/",
            "sec-ch-ua": "'Microsoft Edge';v='113', 'Chromium';v='113', 'Not-A.Brand';v='24'",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "'Windows'",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.35",
            "x-ms-useragent": "AzureOpenAI.Studio/1.0.02315.526",
        }
        print("request = ", request)
        body = json.dumps(request)
        if model_name == utility._MODEL_ADA:
            endpoint = LLMClient._ENDPOINT_ADA
        if model_name == utility._MODEL_BABBAGE:
            endpoint = LLMClient._ENDPOINT_BABBAGE
        if model_name == utility._MODEL_CURIE:
            endpoint = LLMClient._ENDPOINT_CURIE
        if model_name == utility._MODEL_DAVINCI:
            endpoint = LLMClient._ENDPOINT_DAVINCI
        if model_name == utility._MODEL_CHATGPT:
            endpoint = LLMClient._ENDPOINT_CHATGPT
        print("sending request")
        print("endpoint = ", endpoint)
        response = requests.post(endpoint, data=body, headers=headers)
        print("response received= ", response)
        return response.json()
