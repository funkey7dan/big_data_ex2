import math
from pprint import pprint
from itertools import cycle, islice
import numpy


def myData():
    web = [{'URL': 'animals.com', 'tokens': ['jaguar', 'mammal', 'felidae', 'family', 'food'],
            'linksTo': ['pets.org', 'nature.net']},
           {'URL': 'pets.org', 'tokens': ['jaguar', 'cute', 'big', 'cat', 'jaguar', 'my', 'favorite'],
            'linksTo': ['animals.com']},
           {'URL': 'nature.net', 'tokens': ['trees', 'plants', 'flowers', 'nature', 'outdoors'],
            'linksTo': ['animals.com', 'hiking.info']},
           {'URL': 'hiking.info', 'tokens': ['hiking', 'trails', 'outdoors', 'nature'],
            'linksTo': ['nature.net', 'camping.com']},
           {'URL': 'camping.com', 'tokens': ['camping', 'tents', 'outdoors', 'nature'],
            'linksTo': ['hiking.info', 'travel.net']},
           {'URL': 'travel.net', 'tokens': ['travel', 'vacation', 'destinations', 'adventure', 'favorite'],
            'linksTo': ['camping.com', 'airlines.info']},
           {'URL': 'airlines.info', 'tokens': ['flights', 'airlines', 'travel', 'vacation', 'favorite'],
            'linksTo': ['travel.net']},
           {'URL': 'recipes.com', 'tokens': ['recipes', 'cooking', 'food', 'dinner', 'breakfast', 'lunch'],
            'linksTo': ['animals.com']}]
    # web = [{'URL': 'example.com','tokens': ['jaguar','mammal','felidae','family'],'linksTo': ['example.org']},
    #        {'URL': 'example.org','tokens': ['jaguar','cute','big','cat','jaguar','my','favorite'],'linksTo': []},
    #        {'URL': 'example.il','tokens': ['my','cat','big'],'linksTo': ['example.com','example.org']}]
    return web


def mySearchString():
    search_string = ['jaguar', 'family', 'nature', 'travel']
    return search_string


def invertedIndex(data, searchString):
    # The keys of the dictionary will be all the tokens in searchString, and the value for each token will be a list
    # of lists
    out_dict = {s: [] for s in searchString}
    for url_dict in data:
        for t in set(url_dict['tokens']):
            if t in searchString:
                tf = url_dict['tokens'].count(t) / len(url_dict['tokens'])

                occurrences = sum(t in d['tokens'] for d in data)
                idf = math.log2(len(url_dict['tokens']) / occurrences)

                out_dict[t].append([url_dict['URL'], round(tf * idf, 3)])

    # Sort the lists in the inverted index by descending tf-idf score
    for token in out_dict:
        out_dict[token].sort(key=lambda x: x[1], reverse=True)

    return out_dict


def pageRankSimulation(data, numIter, beta):
    # creating matrix
    amount_of_websites = len(data)

    adjacency_matrix = numpy.zeros((amount_of_websites, amount_of_websites))
    for i, website_i in enumerate(data):
        for j, website_j in enumerate(data):
            website_i_outgoing_links = data[i]['linksTo']
            if website_j['URL'] in website_i_outgoing_links:
                # if i -> j
                adjacency_matrix[i, j] = 1 / len(website_i_outgoing_links)

    # creating dump matrix
    dump_matrix = numpy.full((amount_of_websites, amount_of_websites), 1 / amount_of_websites)

    # the matrix we multiply
    A = beta * adjacency_matrix + (1 - beta) * dump_matrix

    importance_vector = numpy.full((amount_of_websites,), 1 / amount_of_websites)
    # performing the iterations
    for i in range(numIter):
        importance_vector = A.dot(importance_vector)

    output_list = []
    for i, website in enumerate(data):
        output_list.append([website['URL'], importance_vector[i]])

    return output_list


def score(tfIdf, pageRankValue):
    # TODO: change
    return sum([tfIdf, pageRankValue])


def roundrobin(*iterables):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)  # .next on Python 2
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def top1(invertedIndex, pageRank):
    seen = set()

    raise NotImplementedError


def main():
    pprint(invertedIndex(myData(), mySearchString()))
    pprint(pageRankSimulation(myData(), 100000, 0.8))


if __name__ == "__main__":
    main()
