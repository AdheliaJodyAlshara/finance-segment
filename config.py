import os
import pandas as pd
from langchain_openai import ChatOpenAI

pd.set_option('display.float_format', '{:.2f}'.format)

url = os.getenv("CSV_URL")
file_id=url.split('/')[-2]
dwn_url='https://drive.google.com/uc?id=' + file_id
df = pd.read_csv(dwn_url)
df_str = df.to_string()

llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0,
    max_tokens=4096,
    # seed=42
)