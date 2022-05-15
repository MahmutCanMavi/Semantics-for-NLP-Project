from nltk.corpus import stopwords
import en_core_web_sm
nlp = en_core_web_sm.load()

def ner_simple(sentence: str):
    """
    Actual function that gives entities as a list
    """
    return [(pair.text) for pair in en_core_web_sm.load()(sentence).ents]

def split_mults(entities : list[str]):
    """
    Some entities consist of multiple words that I split
    """
    return [m.split(' ') for m in entities]

def split_newlines(entities : list[list[str]]):
    """
    Twitter users love using newlines. I don't
    """
    no_new_lines = []
    for entity in entities:
        entity_list = []
        for word in entity:
            for split_word in word.split("\n"): 
                if split_word : entity_list.append(split_word)
        
        no_new_lines.append(entity_list)
    return no_new_lines

def handle_cases(entities : list[list[str]]):
    """
    Handles edge cases in data, e.g. urls, stocks, @s and numbers, hashtags
    """
    edited_entities = []
    for entity in entities:
        for word in entity:
            skip = False
            if word in stopwords.words('english') or word in stopwords.words('german') or word == "the": continue
            for letter in word:
                if letter.isdigit():
                    word = "<number>"
                    break
                elif letter == "@":
                    word = "<user>"
                    break
                elif letter == "$":
                    word = "<money>"
                    break
                elif "https://" in word:
                    word = "<url>"
                    break
                elif letter == "#":
                    skip = True
                    break
            
            if not skip: edited_entities.append(word.lower())
            
    return edited_entities
    
def remove_entities(sentence: str, tokenized_no_sw: list[str]):
	num_ent, url_ent, money_ent, user_ent = False, False, False, False
	for word in handle_cases(split_newlines(split_mults(ner_simple(sentence)))):
		# The remove method removes all, therefore if any of the keywords exist more than 
  		# once an error would occur, hence the barbaric code to handle stuff manually
		if word == "<number>" and num_ent: continue
		elif word == "<user>" and user_ent: continue
		elif word == "<money>" and money_ent: continue
		elif word == "<url>" and url_ent: continue
		if word == "<number>" : num_ent = True
		elif word == "<user>" : user_ent = True
		elif word == "<money>" : money_ent = True
		elif word == "<url>" : url_ent = True
		# For some unknown reason, sometimes stopwords fall through, idk why (very possibly unnecessary)
		if word in stopwords.words('english') or word in stopwords.words('german'): continue
		# BTS fan accounts use unrecognizable characters occasionally, that are sometimes 
  		# classified as entities. I gave up so, if it is removed by the tokenizer, I don't try
		if word in tokenized_no_sw: tokenized_no_sw.remove(word)

	return tokenized_no_sw

if __name__ == '__main__':
    # First english tweet as a test case
    sentence = 'Zimbabwe. Moscow had previously lifted curbs on air links with 15 nations, including the countries of the Eurasian Economic Union (EAEU), Qatar, Mexico, the Dominican Republic, Cuba, UAE, Turkey, Finland, the Czech Republic, Switzerland, South Korea, and Egypt.\n\nUSA AND EU 😂'

    sentence_no_sw_tokenized = ['zimbabwe', '.', 'moscow', 'previously', 'lifted', 'curbs', 'air', 'links', '<number>', 'nations', ',', 'including', 'countries', 'eurasian', 'economic', 'union', '(', 'eaeu', ')', ',', 'qatar', ',', 'mexico', ',', 'dominican', 'republic', ',', 'cuba', ',', 'uae', ',', 'turkey', ',', 'finland', ',', 'czech', 'republic', ',', 'switzerland', ',', 'south', 'korea', ',', 'egypt', '.', 'usa', 'eu', '😂']
    
    print(remove_entities(sentence, sentence_no_sw_tokenized))