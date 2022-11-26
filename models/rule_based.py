import spacy 

nlp = spacy.load('fr_core_news_sm')

doc = nlp(u"Pouvez-vous me dire comment dire «je ne parle pas beaucoup espagnol», en espagnol")
for token in doc:
    print(token, token.lemma_)

