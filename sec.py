from urllib.parse import urlencode

import nest_asyncio
import pandas as pd
import requests
from sec_api import QueryApi

nest_asyncio.apply()

api_key = ""

queryApi = QueryApi(api_key=api_key)

query = {
    "query": 'ticker:AAPL AND formType:"10-Q"',
    "from": "0",
    "size": "1",
    "sort": [{"filedAt": {"order": "desc"}}],
}

response = queryApi.get_filings(query)
df = pd.DataFrame(response["filings"])
