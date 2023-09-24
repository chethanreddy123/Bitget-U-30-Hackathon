import re
import spacy
from ResumeMatcher.scripts.utils.Utils import TextCleaner

nlp = spacy.load('en_core_web_sm')

class DataExtractor:
    def __init__(self, raw_text):
        self.text = raw_text   
        self.clean_text = TextCleaner.clean_text(self.text)  
        self.doc = nlp(self.clean_text)
        
    def extract_keywords(self):
        pos_tags = ['NOUN', 'PROPN']
        keywords = [token.text.lower() for token in self.doc if token.pos_ in pos_tags]  
        return set(keywords)
