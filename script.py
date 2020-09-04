import math
from collections import *
from functools import *
import random
import csv
from numpy import dot

#source: "Data Science from Scratch" by Joel Grus

users_interests = [
    ["mathematics", "theory"],
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regresssion", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistcs", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", 'deep learning', 'Big Data', "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artifical intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", 'regression', 'support vector machines']
]

'''
Manual Curation
recommend the most popular items
'''

popular_interests = Counter(interest
                            for users_interests in users_interests
                            for interest in users_interests).most_common()

# print(popular_interests)

'''
User-Based Collaborative Filtering

Take into account a user's interests and find other users with similar interests
Can't use when there are too many users
'''

#measure similarity
def cosine_similarity(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))

#find the unique interests
unique_interests = sorted(list({interest for user_interests in users_interests
                                for interest in user_interests}))
# print(unique_interests)

#produce interst vector

def make_user_interest_vector(user_interests):
    global unique_interests
    return [1 if interest in user_interests else 0
            for interest in unique_interests]

#create a matrix of user interests

user_interest_matrix = []
for user in users_interests:
    user_interest_matrix.append(make_user_interest_vector(user))

# user_interest_matrix = map(make_user_interest_vector, users_interests)

#compute pairwise similarities between all of our users
# user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
#                       for interest_vector_j in user_interest_matrix]
#                      for interest_vector_i in user_interest_matrix]

user_similarities = []

for interest_vector_i in user_interest_matrix:
    t = []
    for interest_vector_j in user_interest_matrix:
        t.append(cosine_similarity(interest_vector_i, interest_vector_j))
    user_similarities.append(t)

# print(user_similarities[0][9])
# print(user_similarities)

def most_similar_users_to(user_id):
    global unique_interests
    pairs = [(other_user_id, similarity)
             for other_user_id, similarity in
                enumerate(user_similarities[user_id])
             if user_id != other_user_id and similarity > 0]

    return sorted(pairs, key=lambda similarity: similarity[1], reverse=True)

# print(most_similar_users_to(0))

def user_based_suggestions(user_id, include_current_interests=False):
    global unique_interests, users_interests
    #sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity

    #convert them into a sorted list
    suggestions = sorted(suggestions.items(),
                         key=lambda weight: weight[1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(user_based_suggestions(0))

'''
Item-Based Collaborative Filtering
compute the similarities between items directly
'''

#first, transpose the user-intresest matrix so that rows correspond to interests and columns correspond to users

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interest_matrix]
                        for j, _ in enumerate(unique_interests)]

#create the interest similaritiy matrix
interest_similarities = []
for user_vector_i in interest_user_matrix:
    t = []
    for user_vector_j in interest_user_matrix:
        t.append(cosine_similarity(user_vector_i, user_vector_j))
    interest_similarities.append(t)

#find the interest most similar to a topic
def most_similar_interests_to(interest_id):
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
             for other_interest_id, similarity in enumerate(similarities)
             if interest_id != other_interest_id and similarity > 0]

    return sorted(pairs, key=lambda similarity: similarity[1], reverse=True)

# print(most_similar_interests_to(0))

#create recommendations for a user
def item_based_suggestions(user_id, include_current_interests=False):
    global unique_interests, users_interests
    # add up the similarities
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity

    # sort them by weight
    suggestions = sorted(suggestions.items(),
                         key=lambda weight: weight[1],
                         reverse=True)

    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]

print(item_based_suggestions(0))
