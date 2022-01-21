from app import db
from datetime import date
import numpy as np
from bson.objectid import ObjectId
import pickle    
import os 

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


def get_listeners(user_id):
    # Find a specific user and represent it as a list (need only _id, initEmbedding, and birthday)
    listener_dict = ("_id", "initEmbedding", "birthday")
    user = db.users.find({"_id": ObjectId(user_id)})[0]
    listener = list(dict_filter(user, listener_dict).values())
    listener_list = [np.array(listener[:1] + listener[1] + [compute_age(listener[2])]).tolist()]

    # Find all the listeners and represent them as a list
    users = [user for user in db.users.find({"_id": {"$nin": [ObjectId(user_id)]}, "roles": {"$in": ["user"]}})]
    print(users)
    all_listeners = [list(dict_filter(user, listener_dict).values()) for user in users]
    all_listeners = [np.array(user[:1] + user[1] + [compute_age(user[2])]).tolist() for user in all_listeners]
    return listener_list, all_listeners, users


def get_creators(all_listeners, all_users):
    """ 
    Query the data for creators and their followers
    """
    creator_dict = ("_id", "creatorEmbedding")
    creators = [list(dict_filter(user, creator_dict).values()) for user in all_users]
    # [print(creator) for creator in creators if len(creator)==2]
    creators = [creator[0:2] for creator in creators if len(creator)==2]

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
            # 28 inital preferences + 1 age variable
            followers_data = [np.zeros(29)]
        followers_data_full += [np.mean(followers_data, axis=0)]

    # First the creator id, then their embedding, then the average of their followers' initEmbeddings and age.
    creators = [creators[:1] + creators[1] + list(their_followers) for creators, their_followers in zip(creators, followers_data_full)]
    return creators


def get_dataset(user_id):
    # listener_list is the specific user for which we are getting recommendations represented as a list
    listener_list, all_listeners, all_users = get_listeners(user_id)
    # Get all the creators, the data from their transcripts, and the data of their followers
    creators = get_creators(all_listeners, all_users)

    # Create the pairs of our listener with all possible creators, only people with creatorEmbeddings can be creators
    creator_ids = [creator[0] for creator in creators]
    dataset = [[listener_list[0][:1], creator] for creator in creator_ids if listener_list[0][:1] != creator]
    creator_ids = [id[1] for id in dataset]

    # Generate the training dataset
    creators = np.array(creators)
    listeners = np.array(listener_list)
    final_dataset_X = np.array([listeners[:, 1:][0].tolist() + creators[creators[:, 0] == creator, 1:][0].tolist() for _, creator in dataset])

    return creator_ids, final_dataset_X


def clean_recs(item):
    """
    Convert ObjectIds to Strings in order to jsonify afterwards
    """
    item["_id"] = str(item["_id"])
    #TODO: Delete imageProf and audios when they're fully deprecated 
    try:
        item["imageProf"] = str(item["imageProf"])
    except KeyError:
        print("The user" + item["_id"]  + "has no attribute imageProf. In fact, this property is deperecated.")
    try:
        item["avatar"] = str(item["avatar"])
    except KeyError:
        print("The user" + item["_id"]  + "has no attribute avatar")
    try: 
        item["audios"] = [str(audio) for audio in item["audios"]]
    except KeyError:
        print("The user" + item["_id"]  + "has no attribute audios")
    return item


def get_creator_recs(creator_ids, final_dataset_X):   
    # Accessing the model 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + '/models/random_forest.pickle', 'rb') as handle:
        random_forest = pickle.load(handle)

    # Generate predictions for each of the creators
    dtype = [('creator_id', ObjectId), ('prob_to_follow', float)]
    result = np.array(list(zip(np.array(creator_ids), random_forest.predict_proba(final_dataset_X)[:, PROB_INDEX])), dtype=dtype)

    # Sort probs in the descending order but select only 20(could be changed) ObjectIds
    result = np.sort(result, order='prob_to_follow')[::INVERSE_ORDER]
    result = [row[OBJECT_ID_INDEX] for row in result][:NUMBER_OF_CREATORS_TO_RECOMMEND]

    # For those ObjectIds, find all the information in the database and clean it
    creator_recs = list(db.users.find({"_id": {"$in": list(set(result))}}))
    creator_recs = [clean_recs(item) for item in creator_recs]

    # Order them in the right order
    order_dict = {id: index for index, id in enumerate(result)}
    creator_recs = sorted(creator_recs, key=lambda x: order_dict[ObjectId(x["_id"])])
    return creator_recs
