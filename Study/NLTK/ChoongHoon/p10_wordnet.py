from nltk.corpus import wordnet

syns = wordnet.synsets("fun")

#synset
print(syns[0].name())

#just the word
print(syns[0].lemmas()[0].name())

#definition
print(syns[0].definition())

#example
print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("fun"):
    for l in syn.lemmas():
        print("l:",l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("boat.n.01")
#
# print(w1.wup_similarity(w2))
#
# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("car.n.01")
#
# print(w1.wup_similarity(w2))
#
# w1 = wordnet.synset("ship.n.01")
# w2 = wordnet.synset("cat.n.01")
#
# print(w1.wup_similarity(w2))
#
# w1 = wordnet.synset("car.n.01")
# w2 = wordnet.synset("vehicle.n.01")
#
# print(w1.wup_similarity(w2))
#
# word = wordnet.synsets("consequently")
#
# #synset
# print(word[0].name())
#
# #just the word
# print(word[0].lemmas()[0].name())
#
# #definition
# print(word[0].definition())
#
# #example
# print(word[0].examples())
