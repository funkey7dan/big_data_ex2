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


def drawGraph(data):
    import networkx as nx
    import matplotlib.pyplot as plt
    plt.rcParams["figure.figsize"] = (7, 7)
    # Build the graph from the data
    G = nx.DiGraph()
    for entry in data:
        url = entry['URL']
        linksTo = entry['linksTo']
        for neighbor in linksTo:
            G.add_edge(url, neighbor)

    # Draw the graph
    nx.draw(G, with_labels=True, font_weight='bold', node_size=2500, font_size=7, arrowsize=10)
    # plt.show()
    plt.savefig('graph.png')


def mySearchString():
    search_string = ['jaguar', 'family', 'nature', 'travel']
    # search_string = ['jaguar','family']
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
                adjacency_matrix[j, i] = 1 / len(website_i_outgoing_links)

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
    return 0.7 * tfIdf + 0.3 * pageRankValue


def roundrobin(*pairs):
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    num_active = len(pairs)
    nexts = cycle((keyword, iter(it).__next__) for keyword, it in pairs)  # .next on Python 2
    while num_active:
        try:
            for keyword, next in nexts:
                yield keyword, next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def get_score_from_table(table, target_url):
    pr = 0
    for url, s in table[0][1]:
        if target_url == url:
            pr = s

    tfidf_sum = 0
    for keyword, l in table[1:]:
        for pair in l:
            if pair[0] == target_url:
                tfidf_sum += pair[1]
    return score(tfidf_sum, pr)


def get_score_from_list_of_pairs(list_of_pairs, URL: str):
    for pair in list_of_pairs:
        if pair[0] == URL:
            return pair[1]
    return 0


def createTable(invertedIndex, pageRank):
    urls = set()

    for pair in pageRank:
        urls.add(pair[0])

    for keyword, l in invertedIndex.items():
        for pair in l:
            urls.add(pair[0])

    table = [("PageRank", [(url, get_score_from_list_of_pairs(pageRank, url)) for url in urls])]
    for keyword, l in invertedIndex.items():
        table.append((f"tfidf:{keyword}", [(url, get_score_from_list_of_pairs(l, url)) for url in urls]))

    for keyword, l in table:
        l.sort(key=lambda p: p[1], reverse=True)
    return table


def top1(invertedIndex, pageRank):
    seen = set()
    scores = dict()

    keywords = ["PageRank"] + [f"tfidf:{keyword}" for keyword in invertedIndex]
    table = createTable(invertedIndex, pageRank)

    for keyword, (url, _) in roundrobin(*table):
        print(f"sorted access to {url} in {keyword}")

        # checking if in dict
        dict_value = scores.get(url)
        if dict_value:
            # if in dict then add one to counter
            scores[url] = (dict_value[0], dict_value[1] + 1)

            # if we saw as columns then we need to move to seen
            if scores[url][1] == len(keywords):
                seen.add(url)
        else:
            # if not

            # calculating the score
            score = get_score_from_table(table, url)

            # printing the random access (all but keyword)
            temp = keywords[:]
            temp.remove(keyword)
            for k in temp:
                print(f"random access to {url} in {k}")

            # adding to dict
            scores[url] = (score, 1)

        # if seen is 1 then finish
        if len(seen) >= 1:
            break

    # get top 1 from dict
    best_url = max(scores, key=lambda k: scores[k][0])

    return best_url


def main():
    data = myData()
    search_strings = mySearchString()
    drawGraph(data)
    inverted_index = invertedIndex(data, search_strings)
    pprint(inverted_index)
    page_rank = pageRankSimulation(data, 100000, 0.8)
    sorted_pr = sorted(page_rank, key=lambda x: x[1], reverse=True)
    pprint(sorted_pr)
    print(f"best url={top1(inverted_index, page_rank)}")


if __name__ == "__main__":
    main()
