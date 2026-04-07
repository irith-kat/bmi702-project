import os

import requests

BASE_URL = "https://uts-ws.nlm.nih.gov/rest"


def get_code_definition(
    code: str, source: str, api_key: str | None = None
) -> str | None:
    """
    Retrieve human-readable definition for a medical code
    using the UMLS Terminology Services API.

    Args:
        code (str): Medical code (e.g., E11.9, 4548-4, 1049630).
        source (str): Vocabulary name (ICD10CM, LNC, RXNORM).
        api_key (str | None): UMLS API key. Falls back to the UMLS_API_KEY
                              environment variable if not provided.

    Returns:
        str or None
    """
    key = api_key or os.environ.get("UMLS_API_KEY")
    if not key:
        raise ValueError(
            "UMLS API key required. Pass api_key= or set the UMLS_API_KEY environment variable."
        )

    url = f"{BASE_URL}/content/current/source/{source}/{code}"
    r = requests.get(url, params={"apiKey": key})
    if r.status_code != 200:
        return None
    try:
        return r.json()["result"]["name"]
    except KeyError:
        return None
