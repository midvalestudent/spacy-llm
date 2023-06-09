import os

if os.getenv("OPENAI_API_KEY") is None:
    has_openai_key = False
else:
    has_openai_key = True

if (
    os.getenv("AZURE_OPENAI_RESOURCE") is None
    or os.getenv("AZURE_OPENAI_DEPLOYMENT") is None
    or os.getenv("AZURE_OPENAI_API_VERSION") is None
    or os.getenv("AZURE_OPENAI_API_KEY") is None
):
    has_azure_env = False
else:
    has_azure_env = True
