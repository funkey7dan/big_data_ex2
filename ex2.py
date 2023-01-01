import math
from pprint import pprint
from itertools import cycle, islice
import random
import numpy


def myData():
    web = [{'URL': 'animals.com', 'tokens': ['jaguar', 'mammal', 'felidae', 'family', 'food'],
            'linksTo': ['pets.org', 'nature.net']},
           {'URL': 'pets.org', 'tokens': ['jaguar', 'cute', 'big', 'cat', 'jaguar', 'my', 'favorite'],
            'linksTo': ['animals.com']}, {'URL': 'nature.net', 'tokens': ['trees', 'plants', 'flowers', 'nature',
                                                                          'outdoors'],
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


# correct values for example data assuming log10 :
# {'jaguar': [['example.org', 0.05], ['example.com',0.04]], 'family': [['example.com', 0.12]]}

def invertedIndex(data, searchString):
    # The keys of the dictionary will be all the tokens in searchString, and the value for each token will be a list
    # of lists.
    # Each inner list will include two values, a URL and its tf-idf score with respect to this token and
    # document represented by the URL.

    invertedIndex = {s: [] for s in searchString}
    for url_dict in data:
        # build the term frequency dictionary for the current website
        tf_dict = {}
        for t in list(url_dict['tokens']):
            if t in tf_dict:
                tf_dict[t] += 1
            else:
                tf_dict[t] = 1
        for t in searchString:
            if t in tf_dict:
                tf = tf_dict[t] / len(url_dict['tokens'])
                occurrences = sum(t in d['tokens'] for d in data)
                idf = math.log10(len(data) / occurrences)
                invertedIndex[t].append([url_dict['URL'], round(tf * idf, 2)])

    # Sort the lists in the inverted index by descending tf-idf score
    for token in invertedIndex:
        invertedIndex[token].sort(key=lambda x: x[1], reverse=True)

    return invertedIndex


def pageRankSimulation2(data, numIter, beta):
    # creating matrix
    amount_of_websites = len(data)

    adjacency_matrix = numpy.zeros((amount_of_websites, amount_of_websites))
    for i, website_i in enumerate(data):
        website_i_outgoing_links = data[i]['linksTo']
        for j, website_j in enumerate(data):
            if website_j['URL'] in website_i_outgoing_links:
                # if i -> j
                adjacency_matrix[j, i] = 1 / len(website_i_outgoing_links)
    # result = numpy.all((adjacency_matrix == 0),axis = 0)
    # print('Rows that contain only zero:')
    # for i in range(len(result)):
    #     if result[i]:
    #         adjacency_matrix[i,i] = 1
    # adjacency_matrix[0,1] = 1
    # creating dump matrix
    dump_matrix = numpy.full((amount_of_websites, amount_of_websites), 1 / amount_of_websites)

    # the matrix we multiply
    A = beta * adjacency_matrix + (1 - beta) * dump_matrix

    importance_vector = numpy.full((amount_of_websites,), 1 / amount_of_websites)
    # performing the iterations
    for i in range(numIter):
        importance_vector = A.dot(importance_vector)
    importance_vector = numpy.round(importance_vector, 3)

    output_list = []
    for i, website in enumerate(data):
        output_list.append([website['URL'], importance_vector[i]])

    return output_list


def pageRankSimulation(data, numIter, beta):
    freq = {f['URL']: 0 for f in data}

    page = random.choice(data)
    for i in range(numIter):
        freq[page['URL']] += 1
        if page['linksTo'] and random.random() <= beta:
            # if there is next and coin say so, then go to next
            next_str = random.choice(page['linksTo'])
            page = next(i for i in data if i["URL"] == next_str)
        else:
            # if no next or coin said go random
            page = random.choice(data)

    for f in freq:
        freq[f] /= numIter

    output_list = []
    for i, website in enumerate(data):
        output_list.append([website['URL'], round(freq[website['URL']], 3)])

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


def get_score_from_table(target_keyword, table, target_url):
    pr = 0
    if target_keyword != "PageRank":
        print(f"random access to {target_url} in PageRank")
    for url, s in table[0][1]:
        if target_url == url:
            pr = s


    tfidf_sum = 0
    for keyword, l in table[1:]:
        for pair in l:
            if pair[0] == target_url and pair[1] != 0:
                tfidf_sum += pair[1]
                if keyword != target_keyword:
                    print(f"random access to {target_url} in {keyword}")
                break
    return score(tfidf_sum, pr)


def get_score_from_list_of_pairs(list_of_pairs, URL: str):
    for pair in list_of_pairs:
        if pair[0] == URL:
            return pair[1]
    return 0


def createTable(invertedIndex, pageRank):
    urls = set()

    # finding all urls from the table (pretty sure can remove the invertedIndex part)
    for pair in pageRank:
        urls.add(pair[0])

    for keyword, l in invertedIndex.items():
        for pair in l:
            urls.add(pair[0])

    # creating the table
    table = [("PageRank", [(url, get_score_from_list_of_pairs(pageRank, url)) for url in urls])]
    for keyword, l in invertedIndex.items():
        table.append((f"tfidf:{keyword}", [(url, get_score_from_list_of_pairs(l, url)) for url in urls]))

    # sorting
    for keyword, l in table:
        l.sort(key=lambda p: p[1], reverse=True)
    return table


def top1(invertedIndex, pageRank):
    seen = set()
    scores = dict()

    # all keywords we print
    keywords = ["PageRank"] + [f"tf/idf:{keyword}" for keyword in invertedIndex]

    # adding padding (0 for urls that not in columns)
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
            # if not in dict
            # calculating the score
            score = get_score_from_table(keyword, table, url)

            # adding to dict
            scores[url] = (score, 1)

        # if seen is 1 then finish
        if len(seen) >= 1:
            break

    # get top 1 from dict
    best_url = max(scores, key=lambda k: scores[k][0])

    return best_url, round(scores[best_url][0], 3)


def main():
    data = myData()
    search_strings = mySearchString()

    drawGraph(data)

    inverted_index = invertedIndex(data, search_strings)
    pprint(inverted_index)

    page_rank = pageRankSimulation(data, 100000, 0.8)
    sorted_pr = sorted(page_rank, key=lambda x: x[1], reverse=True)
    pprint(sorted_pr)

    best_url, url_score = top1(inverted_index, page_rank)
    print(f"best url is {best_url} with score of {round(url_score, 3)}")


if __name__ == "__main__":
    main()
