# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 23:01:32 2019

@author: ndah
"""

from bs4 import BeautifulSoup

import nltk
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')
from nltk.tokenize.toktok import ToktokTokenizer


from textblob import TextBlob
import re
import unicodedata2 as unicodedata

import spacy
nlp = spacy.load('en')


# contraction map
contraction_mapping = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', 
          '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', 
          '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]


class sentimentPP():

    def __init__(self, corpus):
        self.corpus = corpus
    
    # Remove HTML tags
    def strip_html_tags(self, text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    # Remove accented characters
    def remove_accented_chars(self, text):
        output_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return output_text
    
    # Expand contractions
    def expand_contractions(self, text):   
        contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)
        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match)\
                                    if contraction_mapping.get(match)\
                                    else contraction_mapping.get(match.lower())                       
            expanded_contraction = first_char+expanded_contraction[1:]
            return expanded_contraction
            
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        
        return expanded_text
    

    # remove puntuation
    def remove_punctuations(self, text):
        punc_text = str(text)
        for punct in puncts:
            punc_text = punc_text.replace(punct, f' {punct} ')
        return punc_text
    
    # Remove special characters
    def remove_special_characters(self, text):
        pattern = r'[^a-zA-z0-9\s]'
        clean_text = re.sub(pattern, '', text)
        return clean_text
    
    # Spelling correction
    def spell_checker(self, text):
        return str(TextBlob(text).correct())
    
    # Text lemmatization
    def lemmatize_text(self, text):
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text
    
    # Text stemming
    def simple_stemmer(self, text):
        
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in tokens])
        return text
    
    # Remove stopwords [only considers english language top words]
    def remove_stopwords(self, text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)   
        
        return filtered_text
    
    # main text normalizer
    def normalize_corpus(self, 
                         html_stripping=True, 
                         contraction_expansion=True,
                         accented_char_removal=True, 
                         text_lower_case=True, 
                         spell_check=True,
                         stemming = False,
                         lemmatization=True, 
                         special_char_removal=True, 
                         stopword_removal=True, 
                         remove_digits=True):
        
        #self.__init__(self.corpus)
        
        normalized_corpus = []
        # normalize each document in the corpus
        for doc in self.corpus:
            
            # strip HTML
            if html_stripping:
                doc = self.strip_html_tags(doc)
            
            # Check spelling
            if spell_check:
                doc = self.spell_checker(doc)
                
            # expand contractions    
            if contraction_expansion:
                doc = self.expand_contractions(doc)
                
            # remove accented characters
            if accented_char_removal:
                doc = self.remove_accented_chars(doc)
                
            # lowercase the text    
            if text_lower_case:
                doc = doc.lower()
                
            # remove extra newlines
            #doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
            
            # lemmatize text
            if lemmatization:
                doc = self.lemmatize_text(doc)
                
            if stemming:
                doc = self.simple_stemmer(doc)
                
            # remove special characters and\or digits    
            if special_char_removal:
                # insert spaces between special characters to isolate them    
                special_char_pattern = re.compile(r'([{.(-)!}])')
                doc = special_char_pattern.sub(" \\1 ", doc)
                doc = self.remove_special_characters(doc) 
                
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            
            # remove stopwords
            if stopword_removal:
                doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)
                
            normalized_corpus.append(doc)
            
        return normalized_corpus
    
    
if __name__ == "__main__":
    text = ["It's been said of Dürer's rhino: probably no animal picture has exerted such a profound influence on the arts",
            "When worker bees decide to make a new queen, they feed copious amounts of royal jelly to a few small larvae",
            "triggering a cascade of molecular events resulting in development of a queen",
            "The cause of Napoleon's death has been widely debated. Was it arsenic poisoning? Stomach cancer? A peptic ulcer?",
            "1950 : Early computers cost about as much as a private jet - which would you rather own? ✈️or 🖥",
            "ow that I’m 18 years old, it’s *perfectly* acceptable for me to attend the birthday shindigs of my older", 
            "more distinguished...so, don’t mind me if I humblebrag about being live at #Web30"]
            
    print(text)
    norm = sentimentPP(text)
    print(norm.normalize_corpus(spell_check=False, stemming = True))