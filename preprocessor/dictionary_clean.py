import pandas as pd
import re
from notebooks.happierfuntokenizing import emoticon_string, regex_strings
from notebooks.unicode_codes import EMOTICON_REGEXES, EMOJI_UNICODE, EMOJI_ALIAS_UNICODE, UNICODE_EMOJI

emoticon_regexes = {k.lower(): re.compile(v) for k, v in EMOTICON_REGEXES.items()}
word_re = re.compile(r"""(%s)""" % "|".join(regex_strings), re.VERBOSE | re.I | re.UNICODE)
emoticon_re = re.compile(regex_strings[1], re.VERBOSE | re.I | re.UNICODE)
multSpace = re.compile(r'\s\s+')
startSpace = re.compile(r'^\s+')
endSpace = re.compile(r'\s+$')
multDots = re.compile(r'\.\.\.\.\.+') #more than four periods
newlines = re.compile(r'\s*\n\s*')


class Tokenizer:
    def __init__(self, preserve_case=False, use_unicode=True):
        self.preserve_case = preserve_case
        self.use_unicode = use_unicode

    def tokenize(self, s):
        """
        Argument: s -- any string or unicode object
        Value: a tokenize list of strings; conatenating this list returns the original string if preserve_case=False
        """        
        # Try to ensure unicode:
        if self.use_unicode:
            try:
                s = str(s)
            except UnicodeDecodeError:
                s = str(s).encode('string_escape')
                s = str(s)
        # Fix HTML character entitites:
        #s = self.__html2unicode(s)
        #s = self.__removeHex(s)
        # Tokenize:
        words = word_re.findall(s)
        #print words #debug
        # Possible alter the case, but avoid changing emoticons like :D into :d:
        if not self.preserve_case:            
            words = list(map((lambda x : x if emoticon_re.search(x) else x.lower()), words))
        
        return words       

def removeNumeric(token):
    # remove phone numbers
    token = re.sub(r'((1-\d{3}-\d{3}-\d{4})|(1 \d{3} \d{3} \d{4})|(\d{3} \d{3} \d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))$', '', token)
    # remove all numeric words
    token = '' if token.isdigit() else token
    # remove words with 2 successive digits
    token = '' if re.search(r'\d{2}', token) else token
    return token

def isEmojiOrEmoticon(token):
    if any([v.match(token) for k,v in emoticon_regexes.items()]) or token in UNICODE_EMOJI:
        return True
    else:
        return False

def shrinkSpace(s):
    """turns multiple spaces into 1"""
    s = multSpace.sub(' ',s)
    s = multDots.sub('....',s)
    s = endSpace.sub('',s)
    s = startSpace.sub('',s)
    s = newlines.sub(' <NEWLINE> ',s)
    return s

def prepare_data(ml_options, X_train, X_test, y_train, y_test):
    if ml_options["nlp_dict"] == "LIWC":
        worddict = pd.read_csv("LIWC.csv", sep=";")
        worddict = worddict.iloc[:,0]
        dict_clean = []
        for word in worddict:
            word = str(word)
            if "*" in word:
                dict_clean.append(word[:-1])
            else:
                dict_clean.append(word)

    for nlp_column in ml_options["nlp_columns"]:
        Xtrain_nlp = X_train[["id", nlp_column]]
        Xtrain_nlp.set_index("id", inplace=True)
        Xtest_nlp = X_test[["id", nlp_column]]
        Xtest_nlp.set_index("id", inplace=True)

        Xtrain_nlp = Xtrain_nlp.applymap(lambda s:s.lower() if type(s) == str else s)
        Xtest_nlp = Xtest_nlp.applymap(lambda s:s.lower() if type(s) == str else s)

        for letter in Xtrain_nlp.columns[0]:
            letter = re.split(" ", letter)
        for letter in Xtest_nlp.columns[0]:
            letter = re.split(" ", letter)
        

        
        tokenizer = Tokenizer(preserve_case=False)

    





