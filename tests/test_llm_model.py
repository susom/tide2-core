"""
Comprehensive test suite for LlmModel class.

Tests cover token estimation, provider initialization, response handling,
and error scenarios with proper mocking of cloud services.
"""

from unittest.mock import Mock
from unittest.mock import patch

import pytest

from tide2.utils.llm_model import LlmModel


class TestLlmModelTokenEstimation:
    """Test token estimation functionality."""

    def test_estimate_tokens_from_characters_default_ratio(self):
        """Test token estimation with default 4.0 chars per token ratio."""
        model = LlmModel(project_id="test")

        # Test basic estimation
        assert model.estimate_tokens_from_characters("Hello world!") == 3  # 12 chars / 4 = 3
        assert model.estimate_tokens_from_characters("Test") == 1  # 4 chars / 4 = 1
        assert model.estimate_tokens_from_characters("A") == 1  # minimum 1 token

    def test_estimate_tokens_from_characters_custom_ratio(self):
        """Test token estimation with custom chars per token ratio."""
        model = LlmModel(project_id="test")

        # Test with different ratios
        text = "Hello world!"  # 12 characters
        assert model.estimate_tokens_from_characters(text, chars_per_token=3.0) == 4  # 12/3 = 4
        assert model.estimate_tokens_from_characters(text, chars_per_token=6.0) == 2  # 12/6 = 2
        assert model.estimate_tokens_from_characters(text, chars_per_token=2.0) == 6  # 12/2 = 6

    def test_estimate_tokens_from_characters_edge_cases(self):
        """Test token estimation edge cases."""
        model = LlmModel(project_id="test")

        # Empty string
        assert model.estimate_tokens_from_characters("") == 0

        # None input (type: ignore for testing edge case)
        assert model.estimate_tokens_from_characters(None) == 0  # type: ignore[arg-type]

        # Non-string input (type: ignore for testing edge case)
        assert model.estimate_tokens_from_characters(123) == 0  # type: ignore[arg-type]

        # Very long text
        long_text = "A" * 1000  # 1000 characters
        assert model.estimate_tokens_from_characters(long_text) == 250  # 1000/4 = 250

    def test_estimate_tokens_from_characters_realistic_examples(self):
        """Test token estimation with realistic clinical text examples."""
        model = LlmModel(project_id="test")

        # Short clinical note
        short_note = "Patient John Smith, age 45, presented with chest pain."
        estimated = model.estimate_tokens_from_characters(short_note)
        assert 10 <= estimated <= 20  # Reasonable range

        # Longer clinical note
        long_note = """
        Patient: John Smith
        DOB: 01/15/1978
        MRN: 123456789
        Chief Complaint: Chest pain and shortness of breath
        History: 45-year-old male with history of hypertension presents with acute onset chest pain.
        Physical Examination: Vital signs stable. Heart rate 85 bpm, blood pressure 140/90 mmHg.
        Assessment and Plan: Likely angina. Start aspirin, order EKG and cardiac enzymes.
        """
        estimated = model.estimate_tokens_from_characters(long_note)
        assert 50 <= estimated <= 150  # Reasonable range for longer text


class TestLlmModelTokenConfiguration:
    """Test token configuration functionality."""

    def test_output_tokens_only(self):
        """Test initialization with only output token limits."""
        model = LlmModel(project_id="test-project", max_output_tokens=2048)

        assert model.max_output_tokens == 2048

    def test_token_limits_for_non_google_providers(self):
        """Test that non-Google providers use max_output_tokens for effective max_tokens."""
        model = LlmModel(
            project_id="test-project", provider_type="anthropic", max_output_tokens=1000, model_name="claude-3-sonnet"
        )

        # For non-Google providers, the effective max_tokens should be max_output_tokens
        expected_max_tokens = model._get_effective_max_tokens_for_non_google_providers()
        assert expected_max_tokens == 1000

    def test_effective_max_tokens_method(self):
        """Test the effective max tokens method for non-Google providers."""
        model = LlmModel(
            project_id="test", provider_type="anthropic", max_output_tokens=2000, model_name="claude-3-sonnet"
        )

        # Effective max tokens should equal max_output_tokens for non-Google providers
        assert model._get_effective_max_tokens_for_non_google_providers() == 2000

    @patch("tide2.utils.llm_model.genai")
    def test_get_model_input_token_limit_success(self, mock_genai):
        """Test successful retrieval of model input token limit from Google GenAI API."""
        # Setup mock
        mock_client = Mock()
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 32768
        mock_client.models.get.return_value = mock_model_info
        mock_genai.Client.return_value = mock_client

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-2.0-flash")

        token_limit = model._get_model_input_token_limit()

        assert token_limit == 32768
        mock_client.models.get.assert_called_once_with(model="gemini-2.0-flash")

    @patch("tide2.utils.llm_model.genai")
    def test_get_model_input_token_limit_failure(self, mock_genai):
        """Test handling of API failure when getting model input token limit."""
        # Setup mock to raise an exception
        mock_client = Mock()
        mock_client.models.get.side_effect = Exception("API Error")
        mock_genai.Client.return_value = mock_client

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-2.0-flash")

        token_limit = model._get_model_input_token_limit()

        assert token_limit is None

    @patch("tide2.utils.llm_model.genai")
    def test_get_model_input_token_limit_missing_attribute(self, mock_genai):
        """Test handling when input_token_limit attribute is missing."""
        # Setup mock where model_info doesn't have input_token_limit attribute.
        # Use a spec-restricted mock so getattr(model_info, "input_token_limit", None)
        # reliably returns None instead of auto-creating a child Mock.
        mock_client = Mock()
        mock_model_info = Mock(spec=[])
        mock_client.models.get.return_value = mock_model_info
        mock_genai.Client.return_value = mock_client

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-2.0-flash")

        token_limit = model._get_model_input_token_limit()

        assert token_limit is None

    @patch("tide2.utils.llm_model.genai")
    def test_get_model_input_token_limit_zero_value(self, mock_genai):
        """Test handling when input_token_limit is 0."""
        # Setup mock where input_token_limit is 0
        mock_client = Mock()
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 0
        mock_client.models.get.return_value = mock_model_info
        mock_genai.Client.return_value = mock_client

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-2.0-flash")

        token_limit = model._get_model_input_token_limit()

        assert token_limit is None

    @patch("tide2.utils.llm_model.genai")
    def test_get_model_input_token_limit_none_value(self, mock_genai):
        """Test handling when input_token_limit is None."""
        # Setup mock where input_token_limit is None
        mock_client = Mock()
        mock_model_info = Mock()
        mock_model_info.input_token_limit = None
        mock_client.models.get.return_value = mock_model_info
        mock_genai.Client.return_value = mock_client

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-2.0-flash")

        token_limit = model._get_model_input_token_limit()

        assert token_limit is None

    def test_get_model_input_token_limit_non_google_provider(self):
        """Test that non-Google providers return None for model input token limit."""
        model = LlmModel(project_id="test-project", provider_type="anthropic", model_name="claude-3-sonnet")

        token_limit = model._get_model_input_token_limit()

        assert token_limit is None


class TestLlmModelInitialization:
    """Test LlmModel initialization with different providers."""

    def test_init_google_provider(self):
        """Test initialization with Google provider."""
        model = LlmModel(
            project_id="test-project",
            provider_type="google",
            temperature=0.5,
            max_output_tokens=1500,
            region="us-west1",
            model_name="gemini-pro",
        )

        assert model.project_id == "test-project"
        assert model.provider_type == "google"
        assert model.temperature == 0.5
        assert model.max_output_tokens == 1500
        assert model.region == "us-west1"
        assert model.model_name == "gemini-pro"

    def test_init_openai_provider(self):
        """Test initialization with OpenAI provider."""
        model = LlmModel(
            project_id=12345,  # int project ID
            provider_type="openai",
            endpoint_id=678,
            model_name="gpt-4",
        )

        assert model.project_id == 12345
        assert model.provider_type == "openai"
        assert model.endpoint_id == 678
        assert model.model_name == "gpt-4"

    def test_init_anthropic_provider(self):
        """Test initialization with Anthropic provider."""
        model = LlmModel(project_id="test-project", provider_type="anthropic", model_name="claude-3-sonnet")

        assert model.provider_type == "anthropic"
        assert model.model_name == "claude-3-sonnet"

    def test_init_defaults(self):
        """Test initialization with default values."""
        model = LlmModel(project_id="test")

        assert model.temperature == 0.0
        assert model.max_output_tokens == 1000
        assert model.region == "us-central1"
        assert model.provider_type == "google"
        assert model.system_prompt is None


class TestLlmModelGoogleProvider:
    """Test Google provider functionality with mocking."""

    @patch("tide2.utils.llm_model.genai")
    def test_get_response_google_success(self, mock_genai):
        """Test successful response from Google provider."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Test response from Google"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Create model and get response
        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-pro")

        response = model.get_response("Test prompt", parse_json=False)

        assert response == "Test response from Google"
        mock_client.models.generate_content.assert_called_once()

        # Verify that the generate_content_config was called with max_output_tokens
        call_args = mock_client.models.generate_content.call_args
        config = call_args.kwargs["config"]
        assert hasattr(config, "max_output_tokens") or "max_output_tokens" in str(config)

    @patch("tide2.utils.llm_model.genai")
    def test_get_response_google_with_json_parsing(self, mock_genai):
        """Test Google provider with JSON response parsing."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = '[{"entity": "PATIENT", "text": "John Smith"}]'
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Mock the _parse_response method
        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-pro")

        with patch.object(model, "_parse_response") as mock_parse:
            mock_parse.return_value = [{"entity": "PATIENT", "text": "John Smith"}]

            response = model.get_response("Test prompt", parse_json=True)

            assert response == [{"entity": "PATIENT", "text": "John Smith"}]
            mock_parse.assert_called_once_with('[{"entity": "PATIENT", "text": "John Smith"}]')


class TestLlmModelOpenAIProvider:
    """Test OpenAI provider functionality with mocking."""

    @patch("tide2.utils.llm_model.httpx.Client")
    @patch("tide2.utils.llm_model.auth")
    def test_get_response_openai_success(self, mock_auth, mock_httpx_client):
        """Test successful response from OpenAI provider."""
        # Setup auth mock
        mock_creds = Mock()
        mock_creds.token = "test-token"
        mock_auth.default.return_value = (mock_creds, "test-project")

        # Setup HTTP client mock
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Test response from OpenAI"}}]}
        mock_client_instance.post.return_value = mock_response
        mock_httpx_client.return_value.__enter__.return_value = mock_client_instance

        # Create model and get response
        model = LlmModel(project_id="test-project", provider_type="openai", model_name="gpt-4")

        response = model.get_response("Test prompt", parse_json=False)

        assert response == "Test response from OpenAI"
        mock_client_instance.post.assert_called_once()


class TestLlmModelAnthropicProvider:
    """Test Anthropic provider functionality with mocking."""

    @patch("tide2.utils.llm_model.AnthropicVertex")
    def test_get_response_anthropic_success(self, mock_anthropic):
        """Test successful response from Anthropic provider."""
        # Setup mock - need to properly mock the content block attributes
        mock_client = Mock()
        mock_content = Mock()
        mock_content.text = "Test response from Anthropic"

        # Create a mock that properly handles the hasattr checks in our thread-safe code
        # The code checks: hasattr(content_block, 'text') - this should return True
        # Then checks: hasattr(content_block, 'type') - this should return False for the else branch
        mock_content.configure_mock(**{"text": "Test response from Anthropic"})
        # Mock objects have all attributes, so we need to use spec to limit them
        mock_content = Mock(spec=["text"])  # Only has 'text' attribute, not 'type'
        mock_content.text = "Test response from Anthropic"

        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create model and get response
        model = LlmModel(project_id="test-project", provider_type="anthropic", model_name="claude-3-sonnet")

        response = model.get_response("Test prompt", parse_json=False)

        assert response == "Test response from Anthropic"
        mock_client.messages.create.assert_called_once()

        # Verify that max_tokens parameter uses minimum of input/output tokens
        call_args = mock_client.messages.create.call_args
        # With default values: min(8000, 1000) = 1000
        assert call_args.kwargs["max_tokens"] == 1000

    @patch("tide2.utils.llm_model.AnthropicVertex")
    def test_get_response_anthropic_custom_tokens(self, mock_anthropic):
        """Test Anthropic provider with custom token limits."""
        # Setup mock
        mock_client = Mock()
        mock_content = Mock(spec=["text"])
        mock_content.text = "Custom token test response"
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Create model with custom token limits
        model = LlmModel(
            project_id="test-project", provider_type="anthropic", model_name="claude-3-sonnet", max_output_tokens=500
        )

        response = model.get_response("Test prompt", parse_json=False)

        assert response == "Custom token test response"

        # Verify that max_tokens uses max_output_tokens for non-Google providers
        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["max_tokens"] == 500


class TestLlmModelLlamaProvider:
    """Test LLAMA provider functionality with mocking."""

    @patch("tide2.utils.llm_model.openai.OpenAI")
    @patch("tide2.utils.llm_model.auth")
    def test_get_response_llama_with_token_limits(self, mock_auth, mock_openai):
        """Test LLAMA provider uses minimum of input/output tokens."""
        # Setup auth mock
        mock_creds = Mock()
        mock_creds.token = "test-token"
        mock_auth.default.return_value = (mock_creds, "test-project")

        # Setup OpenAI client mock
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Test LLAMA response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Create model with specific token limits
        model = LlmModel(
            project_id="test-project", provider_type="llama", model_name="llama3-70b", max_output_tokens=800
        )

        # Mock the thread-safe credential method
        with patch.object(model, "_get_authenticated_credentials") as mock_get_creds:
            mock_get_creds.return_value = (mock_creds, "test-token")

            response = model.get_response("Test prompt", parse_json=False)

            assert response == "Test LLAMA response"

            # Verify that max_tokens uses max_output_tokens for non-Google providers
            call_args = mock_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 800


class TestLlmModelMedGemmaProvider:
    """Test MedGemma provider functionality with mocking."""

    @patch("tide2.utils.llm_model.aiplatform")
    def test_get_response_medgemma_success(self, mock_aiplatform):
        """Test successful response from MedGemma provider."""
        # Setup mock
        mock_endpoint = Mock()
        mock_prediction = {"choices": [{"message": {"content": "Test response from MedGemma"}}]}
        mock_response = Mock()
        mock_response.predictions = mock_prediction
        mock_endpoint.predict.return_value = mock_response
        mock_aiplatform.Endpoint.return_value = mock_endpoint

        # Create model and get response
        model = LlmModel(
            project_id="test-project", provider_type="medgemma", endpoint_id=123456789, model_name="medgemma"
        )

        response = model.get_response("Test prompt", parse_json=False)

        assert response == "Test response from MedGemma"
        mock_endpoint.predict.assert_called_once()

        # Verify that max_tokens uses minimum of input/output tokens
        call_args = mock_endpoint.predict.call_args
        instances = call_args.kwargs["instances"]
        # With default values: min(8000, 1000) = 1000
        assert instances[0]["max_tokens"] == 1000


class TestLlmModelErrorHandling:
    """Test error handling scenarios."""

    def test_unsupported_provider_error(self):
        """Test error handling for unsupported provider."""
        model = LlmModel(project_id="test-project", provider_type="unsupported_provider")

        with pytest.raises(ValueError, match="Unsupported provider type"):
            model.get_response("Test prompt")

    @patch("tide2.utils.llm_model.genai")
    def test_google_provider_error(self, mock_genai):
        """Test error handling for Google provider failures."""
        # Setup mock to raise exception
        mock_genai.Client.side_effect = Exception("Google API error")

        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-pro")

        # Should raise exception (current behavior)
        with pytest.raises(Exception, match="Google API error"):
            model.get_response("Test prompt")

    @patch("tide2.utils.llm_model.openai")
    @patch("tide2.utils.llm_model.auth")
    def test_openai_auth_error(self, mock_auth, mock_openai):
        """Test error handling for OpenAI authentication failures."""
        # Setup auth mock to raise exception - need to patch the thread-safe method
        mock_auth.default.side_effect = Exception("Auth error")

        model = LlmModel(project_id="test-project", provider_type="openai", model_name="gpt-4")

        # Mock the thread-safe credential method to raise the auth error
        with patch.object(model, "_get_authenticated_credentials") as mock_creds:
            mock_creds.side_effect = Exception("Auth error")

            # Should raise exception (current behavior)
            with pytest.raises(Exception, match="Auth error"):
                model.get_response("Test prompt")


class TestLlmModelAsyncFunctionality:
    """Test async functionality."""

    @patch("tide2.utils.llm_model.genai")
    def test_get_response_async_method_exists(self, mock_genai):
        """Test that async method exists and can be called."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Async test response"
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        # Create model
        model = LlmModel(project_id="test-project", provider_type="google", model_name="gemini-pro")

        # Check that async method exists
        assert hasattr(model, "get_response_async")
        assert callable(model.get_response_async)


class TestLlmModelJsonParsing:
    """Test JSON response parsing functionality."""

    def test_parse_response_valid_json(self):
        """Test parsing of valid JSON response."""
        model = LlmModel(project_id="test")

        # Test with valid JSON array
        response = '[{"entity": "PATIENT", "text": "John Smith", "confidence": 0.95}]'
        result = model._parse_response(response)

        expected = [{"entity": "PATIENT", "text": "John Smith", "confidence": 0.95}]
        assert result == expected

    def test_parse_response_json_in_text(self):
        """Test parsing JSON embedded in other text."""
        model = LlmModel(project_id="test")

        # Test with JSON embedded in explanation text
        response = """
        Here are the entities I found:
        [{"entity": "PATIENT", "text": "Jane Doe"}]
        Hope this helps!
        """
        result = model._parse_response(response)

        expected = [{"entity": "PATIENT", "text": "Jane Doe"}]
        assert result == expected

    def test_parse_response_no_json(self):
        """Test parsing response with no JSON."""
        model = LlmModel(project_id="test")

        # Test with plain text response
        response = "This is just plain text with no JSON structure."

        # Should raise ValueError (current behavior)
        with pytest.raises(ValueError, match="No JSON array found"):
            model._parse_response(response)

    def test_parse_response_invalid_json(self):
        """Test parsing response with invalid JSON."""
        model = LlmModel(project_id="test")

        # Test with malformed JSON
        response = '[{"entity": "PATIENT", "text": "John Smith"}'  # missing closing bracket

        # Should raise ValueError since no complete JSON array is found
        with pytest.raises(ValueError, match="No JSON array found"):
            model._parse_response(response)


class TestLlmModelIntegration:
    """Integration-style tests with more realistic scenarios."""

    def test_clinical_text_token_estimation_integration(self):
        """Test token estimation with realistic clinical text."""
        model = LlmModel(project_id="test-project", provider_type="google", max_output_tokens=2000)

        # Realistic clinical note
        clinical_text = """
        PATIENT: Smith, John
        DOB: 01/15/1978
        MRN: 123456789
        DATE OF SERVICE: 03/15/2024

        CHIEF COMPLAINT: Chest pain

        HISTORY OF PRESENT ILLNESS:
        45-year-old male with history of hypertension and diabetes mellitus type 2
        presents to the emergency department with acute onset of substernal chest pain
        that started approximately 2 hours ago while mowing the lawn. Pain is described
        as crushing, 8/10 intensity, radiating to the left arm and jaw. Associated
        with diaphoresis and nausea. No shortness of breath. Patient took two doses
        of sublingual nitroglycerin with minimal relief.

        PAST MEDICAL HISTORY:
        1. Hypertension - diagnosed 2015
        2. Diabetes mellitus type 2 - diagnosed 2018
        3. Hyperlipidemia

        MEDICATIONS:
        1. Lisinopril 10mg daily
        2. Metformin 500mg twice daily
        3. Atorvastatin 20mg daily

        PHYSICAL EXAMINATION:
        Vitals: BP 160/95, HR 95, RR 18, Temp 98.6F, O2 Sat 98% on room air
        General: Alert, oriented, appears uncomfortable, diaphoretic
        Cardiovascular: Regular rate and rhythm, no murmurs, rubs, or gallops
        Pulmonary: Clear to auscultation bilaterally

        ASSESSMENT AND PLAN:
        1. Acute coronary syndrome - likely STEMI
           - Obtain 12-lead EKG immediately
           - Serial cardiac enzymes (troponin, CK-MB)
           - Aspirin 325mg chewed
           - Clopidogrel 600mg loading dose
           - Atorvastatin 80mg
           - Cardiology consultation emergent
           - Prepare for possible cardiac catheterization

        2. Hypertension - currently elevated likely secondary to pain/stress
           - Continue home Lisinopril
           - Monitor closely

        3. Diabetes - hold Metformin given potential for contrast exposure

        DISPOSITION: Admit to CCU for monitoring and further management

        Dr. Sarah Johnson, MD
        Emergency Medicine
        """

        estimated_tokens = model.estimate_tokens_from_characters(clinical_text)

        # This clinical note is about 2,100 characters, so roughly 525 tokens
        assert 400 <= estimated_tokens <= 600

        # Test that it's within reasonable bounds for processing
        assert estimated_tokens < model.max_output_tokens  # Should not trigger overflow warning
