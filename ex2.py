import math

def myData():
    web = [{'URL': 'animals.com','tokens': ['jaguar','mammal','felidae','family'],'linksTo': ['pets.org','nature.net']},
           {'URL': 'pets.org','tokens': ['jaguar','cute','big','cat','jaguar','my','favorite'],
            'linksTo': ['animals.com']},{'URL': 'nature.net','tokens': ['trees','plants','flowers','nature','outdoors'],
                                         'linksTo': ['animals.com','hiking.info']},
           {'URL': 'hiking.info','tokens': ['hiking','trails','outdoors','nature'],
            'linksTo': ['nature.net','camping.com']},
           {'URL': 'camping.com','tokens': ['camping','tents','outdoors','nature'],
            'linksTo': ['hiking.info','travel.net']},
           {'URL': 'travel.net','tokens': ['travel','vacation','destinations','adventure'],
            'linksTo': ['camping.com','airlines.info']},
           {'URL': 'airlines.info','tokens': ['flights','airlines','travel','vacation'],'linksTo': ['travel.net']},
           {'URL': 'recipes.com','tokens': ['recipes','cooking','food','dinner','breakfast','lunch'],'linksTo': []}]
    web = [{'URL': 'example.com','tokens': ['jaguar','mammal','felidae','family'],'linksTo': ['example.org']},
           {'URL': 'example.org','tokens': ['jaguar','cute','big','cat','jaguar','my','favorite'],'linksTo': []},
           {'URL': 'example.il','tokens': ['my','cat','big'],'linksTo': ['example.com','example.org']}]
    return web

def mySearchString():
    search_string = ['jaguar','family']
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
                out_dict[t].append([url_dict['URL'],tf * idf])
    # Sort the lists in the inverted index by descending tf-idf score
    for token in out_dict:
        out_dict[token].sort(key = lambda x: x[1],reverse = True)
    return out_dict

print(invertedIndex(myData(),mySearchString()))

