import pickle

scorer = pickle.load(open('scorer.pkl', 'rb'))
print(scorer.get_results())