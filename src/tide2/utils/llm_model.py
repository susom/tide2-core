"""
AgentModel Module

This module provides a unified interface for creating AI agents using different providers
for Vertex AI. It uses two interfaces, one for google model and the other for models deployed in an vertex Ai endpoint.
"""

import asyncio
import json
import logging
import re
import threading

import httpx
import openai
from anthropic import AnthropicVertex
from google import auth
from google import genai
from google.auth.transport.requests import Request
from google.cloud import aiplatform
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


class LlmModel:
    """
    A thread-safe class for creating and managing AI agents with different underlying models.

    This class provides a unified interface for creating agents powered by various providers
    including Google Vertex AI, OpenAI, Anthropic, LLAMA, and MedGemma. It handles authentication,
    model initialization, and agent setup with appropriate configurations in a thread-safe manner.

    Attributes:
        temperature: Model temperature setting for response randomness
        max_output_tokens: Maximum tokens allowed for output
        provider_type: The provider type being used
        model_name: Name of the model
        project_id: Google Cloud project ID
        region: Cloud region for the API
        system_prompt: System prompt to guide agent behavior
        endpoint_id: Vertex AI endpoint ID (for certain providers)
        _auth_lock: Thread lock for synchronizing credential operations
        _credentials_cache: Cached credentials to avoid repeated auth calls
    """

    # Class-level locks for different providers
    _auth_lock = threading.Lock()
    _aiplatform_lock = threading.Lock()
    _credentials_cache = {}
    _aiplatform_initialized = {}

    def __init__(
        self,
        project_id: str | int,
        provider_type: str = "google",
        temperature: float = 0.0,
        max_output_tokens: int = 1000,
        region: str = "us-central1",
        system_prompt: str | None = None,
        model_name: str | None = None,
        endpoint_id: int | None = None,
    ):
        """
        Initialize an AgentModel with the specified provider and configurations.

        Args:
            project_id: Google Cloud project ID (str) for Google provider or project number (int) for OpenAI provider
            system_prompt: System prompt to guide the agent's behavior
            provider_type: Type of provider to use ('google' or 'openai')
            temperature: Model temperature (default: 0.0)
            max_output_tokens: Maximum tokens for output (default: 1000)
            region: Cloud region for the API (default: 'us-central1')
            model_name: Name of the Google model to use (required for Google provider)
            endpoint_id: The Vertex AI endpoint ID (required for OpenAI provider)

        Raises:
            ValueError: If required parameters for the selected provider are missing or invalid
        """

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.provider_type = provider_type
        self.model_name = model_name
        self.project_id = project_id
        self.region = region
        self.system_prompt = system_prompt
        self.endpoint_id = endpoint_id

    def _get_authenticated_credentials(self):
        """
        Thread-safe method to get authenticated credentials.
        Uses caching to avoid repeated authentication calls.

        Returns:
            Tuple of (credentials, access_token)
        """
        cache_key = f"{self.project_id}_{self.region}"

        with self._auth_lock:
            # Check if we have cached credentials
            if cache_key in self._credentials_cache:
                creds, token = self._credentials_cache[cache_key]
                # Check if token is still valid (simple check)
                try:
                    if hasattr(creds, "valid") and creds.valid:
                        return creds, token
                except Exception as e:  # noqa: BLE001
                    # If validation fails, refresh credentials
                    logger.debug("Credential validation check failed, will refresh: %s", e)

            # Get new credentials
            creds, _ = auth.default()
            auth_req = Request()

            # Handle credentials safely with proper type checking
            try:
                # For most Google credentials, try to access the token directly
                token = getattr(creds, "token", None)

                # If no token attribute, try refreshing if the method exists
                if token is None:
                    refresh_method = getattr(creds, "refresh", None)
                    if refresh_method and callable(refresh_method):
                        try:
                            refresh_method(auth_req)
                            token = getattr(creds, "token", None)
                        except Exception as refresh_error:
                            logger.debug(f"Credential refresh failed: {refresh_error}")

                # Final fallback - convert credentials to string
                if token is None:
                    token = str(creds)

            except Exception as e:
                logger.warning(f"Failed to process credentials: {e}")
                token = str(creds)

            # Cache the credentials
            self._credentials_cache[cache_key] = (creds, token)

            return creds, token

    def _initialize_aiplatform_safely(self):
        """
        Thread-safe initialization of aiplatform for the medgemma provider.
        """
        init_key = f"{self.project_id}_{self.region}"

        with self._aiplatform_lock:
            if init_key not in self._aiplatform_initialized:
                aiplatform.init(project=str(self.project_id), location=self.region)
                self._aiplatform_initialized[init_key] = True

    def _get_model_input_token_limit(self) -> int | None:
        """
        Get the input token limit for the current model using the Google GenAI API.

        Returns:
            The input token limit for the model, or None if it cannot be retrieved
        """
        if self.provider_type.lower() != "google" or not self.model_name:
            return None

        try:
            client = genai.Client(
                vertexai=True,
                project=str(self.project_id),
                location=self.region,
            )

            # Get model information
            model_info = client.models.get(model=self.model_name)

            # Check if input_token_limit exists and has a valid value
            token_limit = getattr(model_info, "input_token_limit", None)
            if token_limit is not None and token_limit > 0:
                return token_limit
            logger.info(f"Model {self.model_name} does not have input_token_limit parameter or it is zero/null")
            return None

        except Exception as e:
            logger.warning(f"Failed to get model input token limit for {self.model_name}: {e}")

        return None

    def _get_effective_max_tokens_for_non_google_providers(self) -> int:
        """
        Get the effective max tokens for non-Google providers.
        For non-Google providers, we use a conservative approach and set max_tokens
        to the max_output_tokens value to avoid exceeding model limits.

        Returns:
            The effective max tokens value to use
        """
        # For non-Google providers, we use max_output_tokens as the safe limit
        # since we don't have direct access to model input token limits
        return self.max_output_tokens

    def _parse_response(
        self,
        response: str,
    ) -> list[dict] | None:
        """
        Parse the response from the model.

        Args:
            response: The response object from the model

        Returns:
            The parsed response
        """

        if isinstance(response, str):
            json_match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                json_dict = json.loads(json_str)
            else:
                raise ValueError("No JSON array found in the response.")

            return json_dict
        logger.error("Response is not a string. Expected a string response.")
        raise TypeError("Response is not a string. Expected a string response.")

    def get_response(self, prompt: str, parse_json=True) -> list[dict] | str | None:
        """
        Get a response from the model based on the provider type.

        Args:
            prompt: The input prompt for the model

        Returns:
            The response from the model
        """

        # Initialize provider and model based on provider_type
        if self.provider_type.lower() == "google":
            client = genai.Client(
                vertexai=True,
                project=str(self.project_id),
                location=self.region,
            )

            generate_content_config = genai_types.GenerateContentConfig(
                temperature=self.temperature,
                top_p=0.95,
                seed=0,
                max_output_tokens=self.max_output_tokens,
                safety_settings=[
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=genai_types.HarmBlockThreshold.OFF,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=genai_types.HarmBlockThreshold.OFF,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=genai_types.HarmBlockThreshold.OFF,
                    ),
                    genai_types.SafetySetting(
                        category=genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=genai_types.HarmBlockThreshold.OFF,
                    ),
                ],
                thinking_config=genai_types.ThinkingConfig(
                    thinking_budget=-1,
                ),
            )

            if not self.model_name:
                raise ValueError("model_name is required for Google provider")

            response = client.models.generate_content(
                model=self.model_name,
                contents=[prompt],  # Simplified content format
                config=generate_content_config,
            )

            response_text = response.text

        elif self.provider_type.lower() == "llama":
            MAAS_ENDPOINT = f"{self.region}-aiplatform.googleapis.com"
            base_url = (
                f"https://{MAAS_ENDPOINT}/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi"
            )

            # Thread-safe credential acquisition
            creds, access_token = self._get_authenticated_credentials()

            client = openai.OpenAI(
                base_url=base_url,
                api_key=access_token,
            )

            if not self.model_name:
                raise ValueError("model_name is required for LLAMA provider")

            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt, "type": "text"},
                        ],
                    }
                ],
                max_tokens=self._get_effective_max_tokens_for_non_google_providers(),
            )

            response_text = response.choices[0].message.content

        elif self.provider_type.lower() == "openai":
            # Vertex's OpenAI-compatible endpoint uses a per-region host
            # ({region}-aiplatform.googleapis.com), except for region="global"
            # which is served from the unprefixed aiplatform.googleapis.com host.
            ENDPOINT = (
                "aiplatform.googleapis.com" if self.region == "global" else f"{self.region}-aiplatform.googleapis.com"
            )
            BASE_URL = f"https://{ENDPOINT}/v1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"

            # Thread-safe credential acquisition
            creds, access_token = self._get_authenticated_credentials()

            # Prepare headers
            headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}

            # Prepare request payload
            payload = {"model": self.model_name, "stream": False, "messages": [{"role": "user", "content": prompt}]}

            # Make the request
            try:
                with httpx.Client() as client:
                    response = client.post(BASE_URL, headers=headers, json=payload, timeout=30.0)
                    response.raise_for_status()
                    response = response.json()
                    response_text = response["choices"][0]["message"]["content"]

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                raise e

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                raise e

        elif self.provider_type.lower() == "anthropic":
            if not self.model_name:
                raise ValueError("model_name is required for Anthropic provider")

            client = AnthropicVertex(region=self.region, project_id=str(self.project_id))

            response = client.messages.create(
                max_tokens=self._get_effective_max_tokens_for_non_google_providers(),
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
            )

            # Handle different content types safely
            response_text = ""
            for content_block in response.content:
                # Check for TextBlock type specifically or any object with text attribute
                if hasattr(content_block, "text"):
                    # Check if it has a type attribute and is text type, or just use the text
                    if hasattr(content_block, "type"):
                        if getattr(content_block, "type", None) == "text":
                            response_text = getattr(content_block, "text", "")
                            break
                    else:
                        # For mock objects or other content that just has text attribute
                        response_text = getattr(content_block, "text", "")
                        break
                # Fallback for other text-like content
                elif str(type(content_block)).find("TextBlock") != -1:
                    response_text = getattr(content_block, "text", str(content_block))
                    break

        elif self.provider_type.lower() == "medgemma":
            if not self.endpoint_id:
                raise ValueError("endpoint_id is required for MedGemma provider")

            # Thread-safe aiplatform initialization
            self._initialize_aiplatform_safely()

            endpoints = {}
            endpoints["endpoint"] = aiplatform.Endpoint(
                endpoint_name=str(self.endpoint_id),
                project=str(self.project_id),
                location=self.region,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            instances = [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": messages,
                    "max_tokens": self._get_effective_max_tokens_for_non_google_providers(),
                    "temperature": self.temperature,
                },
            ]

            response = endpoints["endpoint"].predict(instances=instances, use_dedicated_endpoint=True)

            # Safe response parsing for medgemma
            try:
                if hasattr(response, "predictions") and response.predictions:
                    prediction = (
                        response.predictions[0] if isinstance(response.predictions, list) else response.predictions
                    )
                    if isinstance(prediction, dict) and "choices" in prediction:
                        response_text = prediction["choices"][0]["message"]["content"]
                    else:
                        response_text = str(prediction)
                else:
                    response_text = str(response)
            except (KeyError, IndexError, AttributeError) as e:
                logger.error(f"Error parsing medgemma response: {e}")
                response_text = str(response)
        else:
            raise ValueError(f"Unsupported provider type: {self.provider_type}. Supported providers are 'google'.")

        if not parse_json:
            return response_text
        if response_text is not None:
            return self._parse_response(response_text)
        return None

    async def get_response_async(self, prompt: str, parse_json=True) -> list[dict] | str | None:
        """
        Async helper function for get_response that runs the synchronous API call in a thread pool.

        Args:
            prompt: The input prompt for the model
            parse_json: Whether to parse the response as JSON (default: True)

        Returns:
            The response from the model
        """
        # Run the synchronous get_response method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_response, prompt, parse_json)

    def estimate_tokens_from_characters(self, text: str, chars_per_token: float = 4.0) -> int:
        """
        Estimate the number of tokens based on character count.

        This is a simple approximation method that doesn't use any tokenizer.
        The estimation is based on the common rule of thumb that 1 token ≈ 4 characters
        for English text, though this can vary significantly depending on the language,
        text structure, and specific tokenizer used.

        Args:
            text: The input text to estimate tokens for
            chars_per_token: Average number of characters per token (default: 4.0)
                           - For English: typically 3.5-4.5 characters per token
                           - For code: often fewer characters per token (3-4)
                           - For other languages: can vary significantly

        Returns:
            Estimated number of tokens as an integer

        Example:
            >>> model = LlmModel(project_id="test")
            >>> model.estimate_tokens_from_characters("Hello world!")
            3
            >>> model.estimate_tokens_from_characters("Hello world!", chars_per_token=3.0)
            4
        """
        if not text or not isinstance(text, str):
            return 0

        char_count = len(text)
        estimated_tokens = max(1, round(char_count / chars_per_token))

        return estimated_tokens
