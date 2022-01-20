from app import db
from datetime import date
import numpy as np
from bson.objectid import ObjectId
import pickle    
import os 
from sklearn.ensemble import RandomForestClassifier

NUMBER_OF_CREATORS_TO_RECOMMEND = 20
INVERSE_ORDER = -1
OBJECT_ID_INDEX = 0
PROB_INDEX = 1


dict_filter = lambda x, y: dict([(i,x[i]) for i in x if i in set(y)])


def compute_age(birthdate):
    """
    Compute age given the person's birthdate
    """
    try:
        today = date.today()
        age = today.year - birthdate.year - ((today.month, today.day) < (birthdate.month, birthdate.day))
    except:
        age = 0
    return age


def get_listeners():
    """
    Get certain fields from listeners and represent those as a list
    """
    listener_dict = ("_id", "initEmbedding", "birthday")
    users = [user for user in db.users.find({"roles": {"$in": ["user"]}})]

    listeners = [list(dict_filter(user, listener_dict).values()) for user in users]
    listeners = [np.array(user[:1] + user[1] + [compute_age(user[2])]).tolist() for user in listeners]
    return listeners, users


def get_creators(all_listeners, all_users):
    """ 
    Query the data for creators and their followers
    """
    creator_dict = ("_id", "creatorEmbedding")
    creators = [list(dict_filter(user, creator_dict).values()) for user in all_users]
    creators = [creator for creator in creators if len(creator)==2]

    follows_dict = ("from", "to")
    follows = [list(dict_filter(follow, follows_dict).values()) for follow in db.follows.find()]
    followers_data_full = []
    # For each of the creators, get the average embeddinga and age of their followers
    for creator in creators:
        # If follower is following the creator, get ther user_ids
        followers = [follower[0] for follower in follows if follower[1] == creator[0]]
        # Collect all the followers' listening data 
        followers_data = [listener[1:] for listener in all_listeners if listener[0] in followers]
        if len(followers_data) == 0:
            # 28 initial preferences + 1 age variable
            followers_data = [np.zeros(29)]
        followers_data_full += [np.mean(followers_data, axis=0)]

    # First the creator id, then their embedding, then the average of their followers' initEmbeddings and age.
    creators = [creators[:1] + creators[1] + list(their_followers) for creators, their_followers in zip(creators, followers_data_full)]
    return creators, follows


def get_training_dataset(creators, follows, listeners, users):
    """
    Generate pairs of listeners and creators
    """
    # TODO: Get rid of unbalanced data
    user_ids = [ObjectId(user["_id"]) for user in users]
    dataset = [[listener, creator] for listener in user_ids for creator in user_ids if listener != creator]

    creators = np.array(creators, dtype=object)
    listeners = np.array(listeners, dtype=object)

    final_dataset_X = np.array([listeners[listeners[:, 0] == listener, 1:][0].tolist() \
        + creators[creators[:, 0] == creator, 1:][0].tolist() for listener, creator in dataset])
    final_dataset_y = [1 if [listener, creator] in follows else 0 for listener, creator in dataset ]

    return final_dataset_X, final_dataset_y
    

def generate_model(final_dataset_X, final_dataset_y):
    """
    WATCH_OUT: this will overwrite the previous model, and imemdiately start using the new one.
    """
    # Create the model itself
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(final_dataset_X, final_dataset_y)
    print(clf.predict_proba(final_dataset_X))

    # Pickle the model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    pickle.dump(clf, open(dir_path + "/models/random_forest.pickle", 'wb'))


def train_model():
    # Get the listener data
    listeners, users = get_listeners()   

    # Get the creators' data
    creators, follows = get_creators(all_listeners=listeners, all_users=users)

    # Combine them into one dataset
    final_dataset_X, final_dataset_y = get_training_dataset(creators, follows, listeners, users)

    generate_model(final_dataset_X, final_dataset_y)
