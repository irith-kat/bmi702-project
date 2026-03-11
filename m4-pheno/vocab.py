import requests

UMLS_API_KEY = "deb70cf2-7e57-46ef-a3af-fcb5fb6583cb"
BASE_URL = "https://uts-ws.nlm.nih.gov/rest"


def get_code_definition(code: str, source: str):
    """
    Retrieve human-readable definition for a medical code
    using the UMLS Terminology Services API.

    Args:
        code (str): medical code (e.g., E11.9, 4548-4, 1049630)
        source (str): vocabulary name (ICD10CM, LNC, RXNORM)

    Returns:
        str or None
    """

    url = f"{BASE_URL}/content/current/source/{source}/{code}"
    params = {"apiKey": UMLS_API_KEY}
    r = requests.get(url, params=params)
    if r.status_code != 200:
        return None
    data = r.json()
    try:
        return data["result"]["name"]
    except KeyError:
        return None
