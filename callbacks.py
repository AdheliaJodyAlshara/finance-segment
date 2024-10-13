import time
from langchain.callbacks.base import BaseCallbackHandler

def stream_data(response: str):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)


def clean_text_if_needed(text, texts_to_remove=None):
    # Iterate through each text to be removed and replace it in the input text
    if texts_to_remove is None:
        texts_to_remove = ["AI:"]

    for t in texts_to_remove:
        text = text.replace(t, "")
    # Strip any leading or trailing whitespace
    cleaned_text = text.strip()
    return cleaned_text


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token + ""
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            display_function(clean_text_if_needed(self.text))
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")