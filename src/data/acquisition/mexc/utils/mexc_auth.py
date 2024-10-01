# acquisition/mexc/utils/mexc_auth.py

import hmac
import hashlib
from urllib.parse import urlencode

def generate_signature(api_secret: str, params: dict) -> str:
    query_string = urlencode(params)
    return hmac.new(api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()