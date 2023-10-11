from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests
import sys
from tqdm import tqdm
import openai
from termcolor import colored
import time
from utils import read_yaml_file, remove_punctuation, batchify

class LLMClient:
    with open("./auth.txt", "r") as rf:
        _ENDPOINT = rf.readline().strip()
        token = rf.readline().strip()
        authority = rf.readline().strip()
        _SCOPES = [f"api://{token}/access"]

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(
            lambda: open(".llmapi.bin", "w").write(self._cache.serialize())
            if self._cache.has_state_changed
            else None
        )

        self._app = PublicClientApplication(
            LLMClient.token, authority=LLMClient.authority, token_cache=self._cache
        )
        if os.path.exists(".llmapi.bin"):
            self._cache.deserialize(open(".llmapi.bin", "r").read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + token,
            "X-ModelType": model_name,
        }

        body = str.encode(json.dumps(request))
        retried = 0
        while retried < 10:
            try:
                response = requests.post(
                    LLMClient._ENDPOINT, data=body, headers=headers
                )
                # print(response)
                return response.json()["choices"]
            except Exception as e:
                error = str(e)
                print("retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(60)
        # print(response)
        sys.exit(1)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4)
                )

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]


def extract_seconds(text, retried=5):
    words = text.split()
    for i, word in enumerate(words):
        if "second" in word:
            return int(words[i - 1])
    return 60


def form_request(data, type, **kwargs):
    if "davinci" in type:
        request_data = {
            "prompt": data,
            "max_tokens": 1000,
            "top_p": 1,
            "n": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stream": False,
            "logprobs": None,
            "stop": None,
            **kwargs,
        }
    else:
        # assert isinstance(data, str)
        messages_list = []
        messages_list.append({"role": "user", "content": data})
        request_data = {
            "messages": messages_list,
            "max_tokens": 1000,
            "top_p": 0.95,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            **kwargs,
        }
    # print(request_data)
    return request_data


def llm_init(auth_file="", llm_type='davinci', setting="default"):
    auth = read_yaml_file(auth_file)[llm_type][setting]
    openai.api_type = "azure"
    openai.api_base = auth["api_base"]
    openai.api_key = auth["api_key"]
    openai.api_version = auth["api_version"]
    return auth


def llm_query(data, client, type, task, **config):
    hypos = []
    response = None
    model_name = "davinci" if "davinci" in type else "turbo"
    # batch
    if isinstance(data, list):
        batch_data = batchify(data, 20)
        for batch in tqdm(batch_data):
            retried = 0
            request_data = form_request(batch, model_name, **config)
            if client:
                # print(request_data)
                while True:
                    try:
                        response = client.send_request("text-davinci-003", request_data)
                        response = [r["text"] for r in response]
                        break
                    except Exception as e:
                        error = str(e)
                        print("retring...", error)
                        second = extract_seconds(error, retried)
                        retried = retried + 1
                        time.sleep(second)
            elif "davinci" in type:
                # print(request_data)
                while True:
                    try:
                        response = openai.Completion.create(**request_data)
                        response = response["choices"]
                        response = [r["text"] for r in response]
                        break
                    except Exception as e:
                        error = str(e)
                        print("retring...", error)
                        second = extract_seconds(error, retried)
                        retried = retried + 1
                        time.sleep(second)
                    
            else:
                response = []
                for data in batch:
                    request_data = form_request(data, type, **config)
                    while True:
                        try:
                            result = openai.ChatCompletion.create(**request_data)
                            result = result["choices"][0]["message"]["content"]
                            response.append(result)
                        except Exception as e:
                            error = str(e)
                            print("retring...", error)
                            second = extract_seconds(error, retried)
                            retried = retried + 1
                            time.sleep(second)
                        break

            # print(response)
            if task:
                results = [str(r).strip().split("\n\n")[0] for r in response]
            else:
                results = [str(r).strip() for r in response]
            # print(results)
            # results = [str(r['text']).strip() for r in response]
            # print(results)
            hypos.extend(results)
    else:
        retried = 0
        while True:
            try:
                if client:
                    print("querying client...")
                    request_data = form_request(data, "davinci", **config)
                    response = client.send_request("text-davinci-003", request_data)
                    result = response[0]["text"].strip()
                    break
                else:
                    print(type)
                    result = ""
                    if "turbo" in type or 'gpt4' in type:
                        request_data = form_request(data, type, **config)
                        response = openai.ChatCompletion.create(**request_data)
                        # print(response)
                        result = response["choices"][0]["message"]["content"]
                    else:
                        request_data = form_request(data, type=type, **config)
                        response = openai.Completion.create(**request_data)["choices"][
                            0
                        ]["text"]
                        # result = result['text']
                        result = response.strip()
                    break
            except Exception as e:
                error = str(e)
                print("retring...", error)
                second = extract_seconds(error, retried)
                retried = retried + 1
                time.sleep(second)
        if task:
            result = result.split("\n\n")[0]

        hypos = result
    return hypos


def paraphrase(sentence, client, type, **kwargs):
    if isinstance(sentence, list):
        resample_template = [
            f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{s}\nOutput:"
            for s in sentence
        ]

    else:
        resample_template = f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput:{sentence}\nOutput:"
    print(resample_template)
    results = llm_query(resample_template, client, type, False, **kwargs)
    return results


def llm_cls(dataset, client=None, type=None, **config):
    hypos = []
    results = llm_query(dataset, client=client, type=type, task=True, **config)
    if isinstance(results, str):
        results = [results]
    hypos = [remove_punctuation(r.lower()) for r in results]

    return hypos



if __name__ == "__main__":
    # llm_client = LLMClient()
    llm_client = None
    llm_type = 'turbo'
    start = time.time()
    data =  ["""Q: Tom bought a skateboard for $ 9.46 , and spent $ 9.56 on marbles . Tom 
also spent $ 14.50 on shorts . In total , how much did Tom spend on toys ?                                                 
A: Let's think step by step. """]
    config = llm_init(auth_file="auth.yaml", llm_type=llm_type, setting="default")
    para = llm_query(
        data[0], client=llm_client, type=llm_type, task=False, temperature=0, **config
    )
    print(para)
    end = time.time()
    print(end - start)
