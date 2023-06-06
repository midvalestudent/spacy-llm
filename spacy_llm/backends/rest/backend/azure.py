import os
from typing import Any, Dict, Iterable, List, Sized

import requests  # type: ignore[import]
import srsly  # type: ignore[import]
from requests import HTTPError

from .base import Backend


def _getenv_pop(
    config: Dict[Any, Any],
    key: Any,
    env_key: str,
    default: Any = None,
) -> Any:
    env_value = os.getenv(env_key, None)
    if env_value is not None:
        config.pop(key, None)
        return env_value
    if default is None:
        return config.pop(key)
    return config.pop(key, default)


class AzureOpenAIBackend(Backend):
    def __init__(
        self,
        config: Dict[Any, Any],
        strict: bool,
        max_tries: int,
        interval: float,
        max_request_time: float,
    ):
        """Initializes new AzureOpenAIBackend instance.
        config (Dict[Any, Any]): Config passed on to LLM API.
        strict (bool): If True, ValueError is raised if the LLM API returns a
            malformed response (i. e. any kind of JSON or other response object
            that does not conform to the expectation of how a well-formed
            response object from this API should look like). If False, the API
            error responses are returned by __call__(), but no error will be
            raised. Note that only response object structure will be checked,
            not the prompt response text per se.
        max_tries (int): Max. number of tries for API request.
        interval (float): Time interval (in seconds) for API retries in seconds.
            We implement a base 2 exponential backoff at each retry.
        max_request_time (float): Max. time (in seconds) to wait for request to
            terminate before raising an exception.
        """
        self._resource = _getenv_pop(config, "resource", "AZURE_OPENAI_RESOURCE")
        self._deployment = _getenv_pop(config, "deployment", "AZURE_OPENAI_DEPLOYMENT")
        self._api_version = _getenv_pop(
            config, "api_version", "AZURE_OPENAI_API_VERSION"
        )
        self._is_chat = _getenv_pop(config, "is_chat", "AZURE_OPENAI_IS_CHAT", True)

        if self._is_chat:
            config["url"] = (
                f"https://{self._resource}.openai.azure.com/openai/deployments/"
                f"{self._deployment}/chat/completions?api-version={self._api_version}"
            )
        else:
            config["url"] = (
                f"https://{self._resource}.openai.azure.com/openai/deployments/"
                f"{self._deployment}/completions?api-version={self._api_version}"
            )
        self._model = config["model"]

        super().__init__(config, strict, max_tries, interval, max_request_time)

        # base class assumes "model" is needed in api call; this is an error for AzureOpenAI
        self._config.pop("model")

    @property
    def supported_models(self) -> Dict[str, str]:
        """Returns supported models with their endpoints.

        RETURNS (Dict[str, str]): Supported models with their endpoints.
        """
        return {self._model: self._url}

    @property
    def credentials(self) -> Dict[str, str]:
        """Run a health-check to make sure the deployment exists and is
        accessible using the API key.  Return headers dict that includes
        credentials.

        RETURNS (Dict[str, str]): Headers with 'api-key' entry.
        """
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "Could not find the API key to access the Azure OpenAI API. "
                "Ensure you have an API key then make it available as the "
                "environment variable 'AZURE_OPENAI_API_KEY."
            )

        url = (
            f"https://{self._resource}.openai.azure.com/openai/"
            f"deployments/{self._deployment}?api-version={self._api_version}"
        )
        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        r = self.retry(
            call_method=requests.get,
            url=url,
            headers=headers,
            timeout=self._max_request_time,
        )
        if r.status_code != 200:
            raise ValueError(f"Error accessing {url} ({r.status_code}): {r.text}")
        return {"api-key": api_key}

    def __call__(self, prompts: Iterable[str]) -> Iterable[str]:
        headers = {
            **self._credentials,
            "Content-Type": "application/json",
        }
        api_responses: List[str] = []
        prompts = list(prompts)
        url = self._url

        def _request(json_data: Dict[str, Any]) -> Dict[str, Any]:
            r = self.retry(
                call_method=requests.post,
                url=url,
                headers=headers,
                json={**json_data, **self._config},
                timeout=self._max_request_time,
            )
            try:
                r.raise_for_status()
            except HTTPError as ex:
                res_content = srsly.json_loads(r.content.decode("utf-8"))
                # Include specific error message in exception.
                raise ValueError(
                    f"Request to Azure OpenAI API failed: "
                    f"{res_content.get('error', {}).get('message', str(res_content))}"
                ) from ex
            responses = r.json()

            if "error" in responses:
                if self._strict:
                    raise ValueError(f"API call failed: {responses}.")
                else:
                    assert isinstance(prompts, Sized)
                    return {"error": [srsly.json_dumps(responses)] * len(prompts)}

            return responses

        if self._is_chat:
            # The Azure OpenAI API doesn't support batching for /chat/completions yet
            # so we have to send individual requests.
            for prompt in prompts:
                responses = _request(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                if "error" in responses:
                    return responses["error"]

                # Process responses.
                assert len(responses["choices"]) == 1
                response = responses["choices"][0]
                api_responses.append(
                    response.get("message", {}).get(
                        "content", srsly.json_dumps(response)
                    )
                )

        else:
            responses = _request({"prompt": prompts})
            if "error" in responses:
                return responses["error"]
            assert len(responses["choices"]) == len(prompts)

            for response in responses["choices"]:
                if "text" in response:
                    api_responses.append(response["text"])
                else:
                    api_responses.append(srsly.json_dumps(response))

        return api_responses
