import functools
import math
from collections import defaultdict


class Conversation:
    dictionary = set()
    postings = defaultdict(dict)
    # document_frequency = defaultdict(int)
    length = 0.0

    characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"

    def __init__(self, content):
        self.N = content.count('\n')
        self.initialize_terms_and_postings(content)
        # self.initialize_document_frequencies()
        self.initialize_length()

    def initialize_terms_and_postings(self, content):
        terms = self.tokenize(content)
        unique_terms = set(terms)
        self.dictionary = self.dictionary.union(unique_terms)
        for term in unique_terms:
            self.postings[term] = terms.count(term)

    def tokenize(self, content):
        terms = content.lower().split()
        return [term.strip(self.characters) for term in terms]

    # def initialize_document_frequencies(self):
    #     for term in self.dictionary:
    #         self.document_frequency[term] = len(self.postings[term])

    def initialize_length(self):
        l = 0
        for term in self.dictionary:
            l += self.imp(term)**2
        self.length = math.sqrt(l)            

    def imp(self, term):
        if self.postings[term] is not None:
            return self.postings[term]
            # return self.postings[term] * self.inverse_document_frequency(term)
        else:
            return 0.0

    # def inverse_document_frequency(self, term):
    #     if term in self.dictionary:
    #         return math.log(self.N/self.document_frequency[term],2)
    #     else:
    #         return 0.0

    def do_search(self, query):
        score = self.similarity(query)
        # relevant_document_ids = self.intersection(
        #         [set(self.postings[term].keys()) for term in query])
        # if not relevant_document_ids:
        #     print("No documents matched all query terms.")
        # else:
            # scores = sorted([(id,self.similarity(query))
            #                 for id in relevant_document_ids],
            #                 key=lambda x: x[1],
            #                 reverse=True)
        # print("Score: " + str(score))
        return score
            # for (id,score) in scores:
            #     print(str(score)+": "+self.document_filenames[id])
    
    def intersection(self, sets):
        return functools.reduce(set.union, [s for s in sets])

    def similarity(self, query):
        similarity = 0.0
        for term in query:
            if term in self.dictionary:
                # similarity += self.inverse_document_frequency(term)*self.imp(term)
                similarity += self.imp(term)
        similarity = similarity / self.length
        return similarity
    