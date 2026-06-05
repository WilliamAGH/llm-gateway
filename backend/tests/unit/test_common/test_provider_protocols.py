from app.common.provider_protocols import (
    ALIYUN_PROTOCOL,
    ARK_PROTOCOL,
    DEEPSEEK_PROTOCOL,
    GEMINI_PROTOCOL,
    MOONSHOT_PROTOCOL,
    ZHIPU_PROTOCOL,
    get_frontend_protocol_config,
    normalize_frontend_protocol,
    resolve_implementation_protocol,
    uses_dashscope_thinking,
    uses_deepseek_compatible_thinking,
)


def test_gemini_frontend_protocol_config_exists():
    config = get_frontend_protocol_config("gemini")
    assert config.frontend == "gemini"
    assert config.implementation == GEMINI_PROTOCOL
    assert config.base_url == "https://generativelanguage.googleapis.com"


def test_resolve_implementation_protocol_gemini():
    assert resolve_implementation_protocol("gemini") == GEMINI_PROTOCOL


def test_normalize_frontend_protocol_gemini():
    assert normalize_frontend_protocol("GeMiNi") == "gemini"


def test_deepseek_frontend_protocol_config_exists():
    config = get_frontend_protocol_config("deepseek")
    assert config.frontend == "deepseek"
    assert config.implementation == "openai"
    assert config.base_url == "https://api.deepseek.com"


def test_resolve_implementation_protocol_deepseek():
    assert resolve_implementation_protocol("deepseek") == "openai"


def test_normalize_frontend_protocol_deepseek():
    assert normalize_frontend_protocol("DeepSeek") == DEEPSEEK_PROTOCOL


def test_zhipu_frontend_protocol_config_uses_glm_label():
    config = get_frontend_protocol_config("zhipu")
    assert config.frontend == ZHIPU_PROTOCOL
    assert config.implementation == "openai"
    assert config.base_url == "https://open.bigmodel.cn/api/paas/v4"
    assert config.label == "GLM (OpenAI)"


def test_zhipu_uses_deepseek_compatible_thinking():
    assert uses_deepseek_compatible_thinking("zhipu")


def test_aliyun_frontend_protocol_config_uses_dashscope_label():
    config = get_frontend_protocol_config("aliyun")
    assert config.frontend == ALIYUN_PROTOCOL
    assert config.implementation == "openai"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert config.label == "Dashscope (OpenAI)"


def test_aliyun_uses_dashscope_thinking():
    assert uses_dashscope_thinking("aliyun")


def test_moonshot_frontend_protocol_config_uses_kimi_label():
    config = get_frontend_protocol_config("moonshot")
    assert config.frontend == MOONSHOT_PROTOCOL
    assert config.implementation == "openai"
    assert config.base_url == "https://api.moonshot.cn/v1"
    assert config.label == "Kimi (OpenAI)"


def test_moonshot_uses_deepseek_compatible_thinking():
    assert uses_deepseek_compatible_thinking("moonshot")


def test_ark_frontend_protocol_config_exists():
    config = get_frontend_protocol_config("ark")
    assert config.frontend == ARK_PROTOCOL
    assert config.implementation == "openai"
    assert config.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert config.label == "Ark (OpenAI)"


def test_ark_uses_deepseek_compatible_thinking():
    assert uses_deepseek_compatible_thinking("ark")
