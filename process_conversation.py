import functools
import math
from collections import defaultdict

import conversation

question_words = ["address", "name", "birth", "phone", "country", "county", "district", "subcountry", "parish", "village", "community", "gps", "lat", "lon", "coord", "location", "house", "compount", "school", "social", "network", "email", "age", "religion", "occupation", "work", "years", "old"]

def process_conversation(c):
    convo = conversation.Conversation(c)
    return convo.do_search(question_words)
