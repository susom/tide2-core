"""
LLM JSON Recognizer for detecting Protected Health Information (PHI) and other sensitive entities.

This module provides the LlmJsonRecognizer class which uses Large Language Models (LLMs)
to detect entities in clinical text. The recognizer sends structured prompts to LLMs
and parses JSON responses to extract precise entity spans with confidence scores.

Features:
- Token estimation with overflow warnings
- Exact string matching for all entity occurrences
- Automatic retry logic for robustness
- Support for multiple LLM providers
- Flexible entity type detection driven by prompt configuration
- Thread-safe parallel execution without locks (LLM calls can run concurrently)
"""

import json
import logging
import re
import threading
import time
from typing import Any

from presidio_analyzer import AnalysisExplanation
from presidio_analyzer import RecognizerResult
from presidio_analyzer import RemoteRecognizer
from presidio_analyzer.nlp_engine import NlpArtifacts
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import stop_after_delay
from tenacity import wait_exponential

from tide2.utils.llm_model import LlmModel
from tide2.utils.resource_utils import load_llm_prompt
from tide2.utils.span_metrics import resolve_overlapping_spans

logger = logging.getLogger(__name__)

# Known limitation: Large input texts are not automatically split into chunks.
# The recognizer estimates token count and warns on overflow, but does not
# split-and-rejoin. See: https://github.com/susom/tide2/issues


class LlmJsonRecognizer(RemoteRecognizer):
    """
    A thread-safe, robust LLM-based recognizer for detecting Protected Health Information (PHI) entities.

    This recognizer leverages Large Language Models to identify and tag sensitive entities
    in clinical text. It uses JSON-formatted prompts to get structured responses, then
    employs exact string matching to extract precise entity spans for all occurrences.

    Key Features:
    - **True parallel execution**: Multiple threads can call analyze() simultaneously without serialization
    - **Lock-free thread safety**: Achieved through immutable configuration and stateless operations
    - Token estimation with overflow warnings for large inputs
    - Exact string matching for all entity occurrences in text
    - Built-in retry logic with exponential backoff
    - Support for multiple LLM providers (Google, OpenAI, Anthropic, etc.)
    - Flexible entity type support determined by prompt configuration

    Thread Safety Design:
    - No locks required: LlmModel.get_response() is inherently thread-safe
    - All configuration is immutable after initialization
    - Each analyze() call operates on local variables only
    - This enables efficient parallel processing of multiple notes with maximum throughput
    """

    def __init__(
        self,
        project_id: str | int,
        provider_type: str = "google",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        region: str = "us-central1",
        model_name: str | None = None,
        endpoint_id: int | None = None,
        supported_language: str = "en",
        supported_entities: list[str] | None = None,
        name: str = "LlmJsonRecognizer",
        version: str = "1.0",
        max_retries: int = 3,
        prompt_name: str = "phi_detection",
    ):
        """Initialize the thread-safe LLM JSON Recognizer with immutable configuration.

        Args:
            project_id: Google Cloud project ID (str) or project number (int)
            provider_type: Type of provider to use ('google', 'llama', 'openai', 'anthropic', 'medgemma')
            temperature: Model temperature for response generation (default: 0.0)
            max_tokens: Maximum tokens for output - used for overflow warnings (default: 2000)
            region: Cloud region for the API (default: 'us-central1')
            model_name: Name of the model to use (required for most providers)
            endpoint_id: The Vertex AI endpoint ID (required for some providers)
            supported_language: Language code supported by this recognizer (default: "en")
            supported_entities: List of entity types to detect. If None, defaults to the
                entities defined in the prompt config file. Must be a subset of the prompt
                config's supported_entities.
            name: Name of the recognizer instance (default: "LlmJsonRecognizer")
            version: Version of the recognizer (default: "1.0")
            max_retries: Maximum number of retry attempts for failed requests (default: 3)
            prompt_name: Name of the prompt config in resources/llm_prompts/ or path to
                an external prompt directory (default: "phi_detection")
        """
        # Store immutable configuration parameters as private read-only attributes
        self._project_id = project_id
        self._provider_type = provider_type
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._region = region
        self._model_name = model_name
        self._endpoint_id = endpoint_id
        self._max_retries = max_retries

        # Load prompt config and template from resource files
        prompt_config = load_llm_prompt(prompt_name)
        self._prompt_template = prompt_config["prompt_template"]
        prompt_entities = prompt_config["supported_entities"]
        self._prompt_entities = tuple(prompt_entities)

        if not prompt_entities or not all(isinstance(e, str) and e.strip() for e in prompt_entities):
            raise ValueError(
                f"Prompt config '{prompt_name}' has invalid supported_entities: "
                f"expected a non-empty list of non-empty strings, got {prompt_entities!r}"
            )

        # Validate and set supported entities (immutable copy)
        self._supported_entities_list = tuple(supported_entities or prompt_entities)

        # Validate that all requested entities are supported by the prompt config
        unsupported = set(self._supported_entities_list) - set(prompt_entities)
        if unsupported:
            raise ValueError(
                f"Requested entity types {unsupported} are not supported by prompt '{prompt_name}'. "
                f"Available: {prompt_entities}"
            )

        # Initialize the LLM model (thread-safe - no locks needed)
        # LlmModel.get_response() is thread-safe as it creates new clients per call
        # and uses class-level locks only for credential caching
        self._llm_model = LlmModel(
            project_id=project_id,
            provider_type=provider_type,
            temperature=temperature,
            max_output_tokens=max_tokens,
            region=region,
            system_prompt="",
            model_name=model_name,
            endpoint_id=endpoint_id,
        )

        super().__init__(
            supported_entities=list(self._supported_entities_list),
            supported_language=supported_language,
            name=f"{name}-{model_name}",
            version=version,
        )

    # Thread-safe immutable properties to access configuration
    @property
    def project_id(self) -> str | int:
        """Get the immutable project ID."""
        return self._project_id

    @property
    def provider_type(self) -> str:
        """Get the immutable provider type."""
        return self._provider_type

    @property
    def temperature(self) -> float:
        """Get the immutable temperature."""
        return self._temperature

    @property
    def max_tokens(self) -> int:
        """Get the immutable max tokens."""
        return self._max_tokens

    @property
    def region(self) -> str:
        """Get the immutable region."""
        return self._region

    @property
    def model_name(self) -> str | None:
        """Get the immutable model name."""
        return self._model_name

    @property
    def endpoint_id(self) -> int | None:
        """Get the immutable endpoint ID."""
        return self._endpoint_id

    @property
    def max_retries(self) -> int:
        """Get the immutable max retries."""
        return self._max_retries

    @property
    def supported_entities_list(self) -> list[str]:
        """Get the immutable supported entities list."""
        return list(self._supported_entities_list)  # Return a copy to maintain immutability

    @property
    def llm_model(self) -> LlmModel:
        """Get the thread-safe LLM model instance."""
        return self._llm_model

    def load(self) -> None:
        """Load the recognizer. No special loading required for LLM-based recognizer."""
        pass

    def get_supported_entities(self) -> list[str]:
        """
        Return supported entities by this recognizer.

        Returns:
            List of the supported entity types.
        """
        return self.supported_entities_list

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        """
        Thread-safe analysis of text using LLM to identify and extract entities.

        This method is thread-safe without explicit locks because:
        1. LlmModel.get_response() is inherently thread-safe (creates new clients per call)
        2. All operations use immutable configuration or local variables
        3. Multiple threads can now call analyze() in parallel for better performance

        Args:
            text: The text to analyze for entities
            entities: List of entity types to look for
            nlp_artifacts: Not used by this recognizer

        Returns:
            List of RecognizerResult objects containing detected entities
        """
        start_time = time.time()
        thread_name = threading.current_thread().name
        logger.info(f"[{thread_name}] LlmJsonRecognizer starting analysis")

        if not text.strip():
            return []

        # Filter entities to only those supported by this recognizer
        filter_start = time.time()
        requested_entities = [e for e in entities if e in self.supported_entities_list]
        filter_time = time.time() - filter_start
        logger.debug(f"[{thread_name}] Entity filtering took {filter_time * 1000:.2f}ms")

        if not requested_entities:
            return []

        try:
            # Get LLM response and parse entities with retry logic
            # This is now truly parallel - no serialization!
            results = self.get_response(text, requested_entities)

        except Exception:
            logger.exception("Error in LlmJsonRecognizer.analyze after retries")
            raise

        else:
            elapsed_time = time.time() - start_time
            logger.info(
                f"[{thread_name}] LlmJsonRecognizer completed analysis in {elapsed_time:.3f}s, found {len(results)} entities"
            )
            return results

    def get_response(self, text: str, requested_entities: list[str]) -> list[RecognizerResult]:
        """
        Get LLM response and parse JSON entities with built-in retry logic.

        Sends text to LLM for entity detection, then parses the JSON response.
        Includes automatic retry logic with exponential backoff for robustness.
        Also performs token estimation and warns if prompt may exceed output limits.

        Args:
            text: Input text to analyze for entities
            requested_entities: List of entity types to include in results

        Returns:
            List of RecognizerResult objects containing detected entities

        Raises:
            Exception: If all retry attempts fail
        """

        @retry(
            stop=stop_after_attempt(self.max_retries) | stop_after_delay(180),
            wait=wait_exponential(multiplier=1.5, min=5, max=60),
            retry=retry_if_exception_type((Exception,)),
            reraise=True,
        )
        def _get_llm_response_and_parse_with_retry(text: str, requested_entities: list[str]) -> list[RecognizerResult]:
            """
            Get LLM response and parse JSON entities with retry logic.

            This method combines the LLM request and JSON parsing into a single
            operation with retry capabilities. If either the LLM request fails
            or the JSON parsing fails, the entire operation will be retried.

            Args:
                text: Input text to analyze
                requested_entities: List of entity types to include in results

            Returns:
                List of RecognizerResult objects

            Raises:
                Exception: If all retry attempts fail
            """
            # Get LLM response
            llm_response = self._get_llm_response_internal(text)
            if not llm_response:
                raise ValueError("LLM returned empty response")
            # Parse JSON and extract entities
            return self._parse_json_response_internal(llm_response, text, requested_entities)

        return _get_llm_response_and_parse_with_retry(text, requested_entities)

    def _get_llm_response_internal(self, text: str) -> str:
        """
        Thread-safe method to send text to LLM and get JSON response.

        Estimates the token count of the input prompt and warns if it may exceed
        the model's maximum output token limit, which could result in incomplete responses.

        Thread safety is achieved without locks because:
        1. LlmModel.get_response() is inherently thread-safe
        2. estimate_tokens_from_characters() only reads immutable configuration
        3. All operations use local variables or immutable instance attributes

        This allows true parallel execution of multiple LLM API calls!

        Args:
            text: Input text to analyze

        Returns:
            LLM response string with JSON content

        Raises:
            Exception: If LLM request fails
        """
        thread_name = threading.current_thread().name
        try:
            # Create the formatted prompt
            prompt_start = time.time()
            formatted_prompt = self._prompt_template.replace("{clinical_text}", text)
            prompt_time = time.time() - prompt_start
            logger.debug(f"[{thread_name}] Prompt formatting took {prompt_time * 1000:.2f}ms")

            # Estimate token count and warn if it may exceed max_tokens
            # This is thread-safe as it only reads immutable configuration
            token_start = time.time()
            estimated_tokens = self.llm_model.estimate_tokens_from_characters(formatted_prompt)
            token_time = time.time() - token_start
            logger.debug(f"[{thread_name}] Token estimation took {token_time * 1000:.2f}ms ({estimated_tokens} tokens)")

            if estimated_tokens > self.max_tokens:
                logger.warning(
                    f"Input prompt estimated at {estimated_tokens} tokens exceeds max_tokens "
                    f"({self.max_tokens}). Response may be incomplete or truncated."
                )

            # Get response from LLM (not parsing as JSON yet, we want raw text to validate)
            # This is the key operation that can now run in parallel across threads!
            api_start = time.time()
            response = self.llm_model.get_response(formatted_prompt, parse_json=False)
            api_time = time.time() - api_start
            logger.info(f"[{thread_name}] LLM API call took {api_time:.3f}s")

            # Validate response
            validate_start = time.time()
            if isinstance(response, str):
                stripped_response = response.strip()
                if not stripped_response:
                    raise ValueError("LLM returned empty response")
                validate_time = time.time() - validate_start
                logger.debug(f"[{thread_name}] Response validation took {validate_time * 1000:.2f}ms")
                return stripped_response
            raise TypeError(f"Unexpected response type from LLM: {type(response)}")

        except Exception:
            logger.exception("Error getting LLM response")
            raise

    def _parse_json_response_internal(
        self, json_response: str, original_text: str, requested_entities: list[str]
    ) -> list[RecognizerResult]:
        """
        Parse JSON response from LLM to extract entity information using exact string matching.

        The strategy:
        1. Parse JSON into entity dictionary with types, texts, and confidence scores
        2. For each entity text, find all occurrences in the original text using exact string matching
        3. Create a RecognizerResult for each occurrence found
        4. Validate entity types against prompt config entities and requested_entities

        Args:
            json_response: JSON-formatted response from LLM with entity information
            original_text: Original input text for span validation and matching
            requested_entities: List of entity types to include in results

        Returns:
            List of RecognizerResult objects with detected entities

        Raises:
            Exception: If JSON parsing or entity extraction fails
        """
        thread_name = threading.current_thread().name
        parse_total_start = time.time()

        try:
            # Parse JSON
            json_start = time.time()
            try:
                # Strip markdown code blocks if present
                json_text = json_response.strip()
                if json_text.startswith("```"):
                    # Remove opening ```json or ```
                    lines = json_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    # Remove closing ```
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    json_text = "\n".join(lines)

                entities_dict = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.exception("Failed to parse JSON response")
                logger.debug(
                    f"Problematic JSON (after stripping): {json_text if 'json_text' in locals() else json_response}"
                )
                raise ValueError(f"Invalid JSON response from LLM: {e}") from e

            json_time = time.time() - json_start
            logger.debug(f"[{thread_name}] JSON parsing took {json_time * 1000:.2f}ms")

            # Validate JSON structure
            if not isinstance(entities_dict, dict):
                raise TypeError(f"Expected JSON object (dict), got {type(entities_dict)}")

            # If empty dict, return empty results
            if not entities_dict:
                logger.info("No entities found in JSON response")
                return []

            # Extract entities with exact string matching
            match_start = time.time()
            results = self._extract_entities_with_exact_matching(entities_dict, requested_entities, original_text)
            match_time = time.time() - match_start
            logger.debug(f"[{thread_name}] Entity matching took {match_time * 1000:.2f}ms")

            parse_total_time = time.time() - parse_total_start
            logger.info(f"[{thread_name}] JSON parsing & entity extraction took {parse_total_time:.3f}s")

        except Exception:
            logger.exception("Error parsing JSON response")
            logger.debug(f"Problematic JSON: {json_response}")
            raise

        else:
            return results

    def _extract_entities_with_exact_matching(
        self, entities_dict: dict[str, Any], requested_entities: list[str], original_text: str
    ) -> list[RecognizerResult]:
        """
        Extract entities using exact string matching with non-maximum suppression.

        For each entity in the JSON response:
        1.  Deduplicate entities from the LLM output.
        2.  Find all occurrences of each unique entity text in the original text.
        3.  Resolve overlapping spans by keeping the longest one.
        4.  Create a RecognizerResult for each non-overlapping, longest span.

        Args:
            entities_dict: Dictionary from JSON with entity types as keys and entity lists as values
            requested_entities: Entities requested by analyzer
            original_text: The original text to search in

        Returns:
            List of RecognizerResult objects
        """
        all_spans = []

        # 1. Collect all unique entities and their occurrences
        unique_entities = {}
        for entity_type, entities_list in entities_dict.items():
            if entity_type not in self._prompt_entities or entity_type not in requested_entities:
                continue
            if not isinstance(entities_list, list):
                continue

            for entity in entities_list:
                if not isinstance(entity, dict) or "text" not in entity or "confidence" not in entity:
                    continue

                entity_text = entity["text"].strip()
                if not entity_text:
                    continue

                # Store under a tuple of (text, type) to handle cases where the same text is tagged with different types
                entity_key = (entity_text, entity_type)
                if entity_key not in unique_entities:
                    unique_entities[entity_key] = entity["confidence"]

        # 2. Find all occurrences of each unique entity
        for (entity_text, entity_type), confidence in unique_entities.items():
            escaped_pattern = re.escape(entity_text)
            for match in re.finditer(escaped_pattern, original_text):
                all_spans.append(
                    {
                        "start": match.start(),
                        "end": match.end(),
                        "entity_type": entity_type,
                        "score": confidence,
                        "span_text": entity_text,
                    }
                )

        # 3. Resolve overlapping spans using non-maximum suppression
        if not all_spans:
            return []

        final_spans = resolve_overlapping_spans(all_spans)

        # 4. Create RecognizerResult objects from the final non-overlapping spans
        results: list[RecognizerResult] = []
        for span in final_spans:
            explanation = AnalysisExplanation(
                recognizer=self.__class__.__name__,
                original_score=span["score"],
                textual_explanation=f"LLM identified {span['entity_type']} entity with confidence {span['score']}",
                pattern=f"exact-match {span['entity_type']}",
            )
            results.append(
                RecognizerResult(
                    entity_type=span["entity_type"],
                    start=span["start"],
                    end=span["end"],
                    score=span["score"],
                    analysis_explanation=explanation,
                )
            )

        return results
