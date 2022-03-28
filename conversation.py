import functools
import math
from collections import defaultdict


class Conversation:
    """Class to represent a conversation and contains the logic to determine the cosine similarity to this and a query

    Returns:
        Conversation: this
    """
    dictionary = set()
    postings = defaultdict(dict)
    length = 0.0

    characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"

    def __init__(self, content):
        """Constructs new instance of Conversation

        Args:
            content (string): The conversation data
        """
        self.N = content.count('\n')
        self.initialize_terms_and_postings(content)
        self.initialize_length()

    def initialize_terms_and_postings(self, content):
        """Tokenizes the terms in this conversation and builds a dictionary based on them. 
            It then determines that count of each word in the dictionary

        Args:
            content (string): The conversation data
        """
        terms = self.tokenize(content)
        unique_terms = set(terms)
        self.dictionary = self.dictionary.union(unique_terms)
        for term in unique_terms:
            self.postings[term] = terms.count(term)

    def tokenize(self, content):
        """Splits and strips a string into individual words

        Args:
            content (string): The content to tokenize

        Returns:
            array<String>: The tokenized content
        """
        terms = content.lower().split()
        return [term.strip(self.characters) for term in terms]

    def initialize_length(self):
        """Determine the Euclidean length of this conversation
        """
        l = 0
        for term in self.dictionary:
            l += self.imp(term)**2
        self.length = math.sqrt(l)            

    def imp(self, term):
        """Determine the importance of a term to this conversation. Originally used inverse document freq as well, but as this is a single conversation that has been removed

        Args:
            term (string): The term to determine the importance of

        Returns:
            float: The importance of the passed in word to this conversation
        """
        if self.postings[term] is not None:
            return self.postings[term]
        else:
            return 0.0

    def do_search(self, query):
        """Return the cosine similarity between the passed in query and this document

        Args:
            query (array<String>): Array of words to query

        Returns:
            float: The cosine similarity between the passed in query and this document
        """
        similarity = 0.0
        for term in query:
            if term in self.dictionary:
                similarity += self.imp(term)
        similarity = similarity / self.length
        return similarity
    