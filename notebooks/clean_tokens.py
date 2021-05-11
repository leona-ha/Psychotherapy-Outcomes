#!/usr/bin/env python

# Needs Python 3

# example usage:
# the command below assumes the token is in column 3 (0-indexed)
# python clean_tokens.py file_to_clean.tsv
# the command below explicitly sets the token column to 5
# python clean_tokens.py file_to_clean.tsv column_number_of_token 

# file_to_clean.tsv 
# this is a tsv, where the token column is not quoted

# column_number_of_token
# this is the 0-indexed column ofthe tsv where the token lives

# writes data to file_to_clean_cleaned.tsv
# that is: original file name + "_cleaned" + original extension

import sys
import os
import csv
import re
from unicode_codes import UNICODE_EMOJI, SMS_SLANG, EMOTICON_REGEXES, FINAL_TOKENS
from happierfuntokenizing import Tokenizer 

if len(sys.argv) < 2:
	print("You must supply a file to clean: python clean_tokens.py file_to_clean.tsv")
	sys.exit(1)
elif len(sys.argv) > 2:
	token_column = sys.argv[2]
else:
	token_column = 3

path_to_file_to_clean = sys.argv[1]
extension = path_to_file_to_clean.split(".")[-1]
path_to_file_to_write = path_to_file_to_clean.replace("." + extension, "_cleaned." + extension)
path_to_token_file = os.path.dirname(os.path.abspath(__file__)) + "/final_tokens.txt"

emoticon_regexes = {k.lower(): re.compile(v) for k, v in EMOTICON_REGEXES.items()}
tokenizer = Tokenizer(use_unicode=True)

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

def isSMSSlang(token):
	if token in SMS_SLANG:
		return True
	else:
		return False

multSpace = re.compile(r'\s\s+')
startSpace = re.compile(r'^\s+')
endSpace = re.compile(r'\s+$')
multDots = re.compile(r'\.\.\.\.\.+') #more than four periods
newlines = re.compile(r'\s*\n\s*')

def shrinkSpace(s):
	"""turns multiple spaces into 1"""
	s = multSpace.sub(' ',s)
	s = multDots.sub('....',s)
	s = endSpace.sub('',s)
	s = startSpace.sub('',s)
	s = newlines.sub(' <NEWLINE> ',s)
	return s

with open(path_to_file_to_write, "w") as csv_file:
	writer = csv.writer(csv_file, delimiter="\t")
	with open(path_to_file_to_clean) as f:
		reader = csv.reader(f, delimiter="\t")
		for line in reader:
			cleaned_token = shrinkSpace(line[token_column].strip())
			words = tokenizer.tokenize(cleaned_token)
			
			for w in words:
				if removeNumeric(w) and (isEmojiOrEmoticon(w) or isSMSSlang(w.lower()) or w.lower() in FINAL_TOKENS):
					line[token_column] = w
					writer.writerow(line)





			



