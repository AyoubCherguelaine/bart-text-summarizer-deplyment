from transformers import AutoTokenizer, BartForConditionalGeneration
from . import model


def downloadModel():
    modelBart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizerBart = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model.info.state=1
    return (modelBart, tokenizerBart)
