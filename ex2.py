import math
from pprint import pprint

def myData():
    web = [{'URL': 'animals.com','tokens': ['jaguar','mammal','felidae','family','food'],
            'linksTo': ['pets.org','nature.net']},
           {'URL': 'pets.org','tokens': ['jaguar','cute','big','cat','jaguar','my','favorite'],
            'linksTo': ['animals.com']},{'URL': 'nature.net','tokens': ['trees','plants','flowers','nature','outdoors'],
                                         'linksTo': ['animals.com','hiking.info']},
           {'URL': 'hiking.info','tokens': ['hiking','trails','outdoors','nature'],
            'linksTo': ['nature.net','camping.com']},
           {'URL': 'camping.com','tokens': ['camping','tents','outdoors','nature'],
            'linksTo': ['hiking.info','travel.net']},
           {'URL': 'travel.net','tokens': ['travel','vacation','destinations','adventure','favorite'],
            'linksTo': ['camping.com','airlines.info']},
           {'URL': 'airlines.info','tokens': ['flights','airlines','travel','vacation','favorite'],
            'linksTo': ['travel.net']},
           {'URL': 'recipes.com','tokens': ['recipes','cooking','food','dinner','breakfast','lunch'],
            'linksTo': ['animals.com']}]
    # web = [{'URL': 'example.com','tokens': ['jaguar','mammal','felidae','family'],'linksTo': ['example.org']},
    #        {'URL': 'example.org','tokens': ['jaguar','cute','big','cat','jaguar','my','favorite'],'linksTo': []},
    #        {'URL': 'example.il','tokens': ['my','cat','big'],'linksTo': ['example.com','example.org']}]
    return web

def mySearchString():
    search_string = ['jaguar','family','nature','travel']
    return search_string

def invertedIndex(data,searchString):
    # The keys of the dictionary will be all the tokens in searchString, and the value for each token will be a list
    # of lists
    out_dict = {s: [] for s in searchString}
    for url_dict in data:
        for t in set(url_dict['tokens']):
            if t in searchString:
                occurences = sum(t in d['tokens'] for d in data)
                tf = url_dict['tokens'].count(t) / len(url_dict['tokens'])
                idf = math.log2(len(url_dict['tokens']) / occurences)
                out_dict[t].append([url_dict['URL'],round(tf * idf,3)])
    # Sort the lists in the inverted index by descending tf-idf score
    for token in out_dict:
        out_dict[token].sort(key = lambda x: x[1],reverse = True)
    return out_dict

pprint(invertedIndex(myData(),mySearchString()))

def pageRankSimulation(data, numIter, beta):
    raise NotImplementedError

def score(tfIdf, pageRankValue):
    raise NotImplementedError

def top1(invertedIndex, pageRank):
    raise NotImplementedError