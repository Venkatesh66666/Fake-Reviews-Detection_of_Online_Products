import os
import re
import pickle
import shutil
import warnings
from typing import Dict, Any

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_ZIP = os.path.join(BASE_DIR, "models.zip")
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_DATA_DIR)

categories = ["Apparel", "Automotive", "Baby", "Beauty", "Books", "Camera", "Electronics", "Furniture", "Grocery", "Health & Personal Care", "Home", "Home Entertainment", "Home Improvement", "Jewelry", "Kitchen", "Lawn and Garden", "Luggage", "Musical Instruments", "Office Products", "Outdoors", "PC", "Pet Products", "Shoes", "Sports", "Tools", "Toys", "Video DVD", "Video Games", "Watches", "Wireless"]
categories_str = "Apparel"
for i in range (1, len(categories)) :
    categories_str += ", " + categories[i]


def ensure_nltk_resources():
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    resources = (
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    )
    for resource_path, name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(name, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception:
                warnings.warn(
                    "Unable to download NLTK resource '{}' automatically.".format(name),
                    RuntimeWarning,
                )
    # Backward/forward compatibility across NLTK tagger resource names.
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        try:
            nltk.download("averaged_perceptron_tagger_eng", download_dir=NLTK_DATA_DIR, quiet=True)
        except Exception:
            pass


def ensure_model_artifacts():
    if os.path.isdir(MODELS_DIR):
        return
    if os.path.exists(MODELS_ZIP):
        shutil.unpack_archive(MODELS_ZIP, MODELS_DIR, "zip")
        return
    raise FileNotFoundError(
        "Model artifacts not found. Expected '{}' or '{}'.".format(MODELS_DIR, MODELS_ZIP)
    )


def load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_artifacts():
    ensure_model_artifacts()
    ensure_nltk_resources()
    return {
        "countvectorizer": load_pickle("countvectorizer.sav"),
        "labelencoder_1": load_pickle("labelencoder_1.sav"),
        "labelencoder_2": load_pickle("labelencoder_2.sav"),
        "labelencoder_3": load_pickle("labelencoder_3.sav"),
        "ct1": load_pickle("columntransformer1.sav"),
        "ct2": load_pickle("columntransformer2.sav"),
        "ct3": load_pickle("columntransformer3.sav"),
        "bernoullinb": load_pickle("bernoullinb.sav"),
    }


ARTIFACTS = None


def get_artifacts():
    global ARTIFACTS
    if ARTIFACTS is None:
        ARTIFACTS = load_artifacts()
    return ARTIFACTS

def clean_review(review):
    ensure_nltk_resources()
    review = re.sub('[^a-zA-Z]',' ', review)
    review = review.lower()
    review = review.split()
    #print (review)
    review = [word for word in review if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    return review

def countvectorize(statement):
    countvectorizer = get_artifacts()["countvectorizer"]
    statement = countvectorizer.transform(statement).toarray()
    return statement


def onehotencode(rating, verified_purchase, product_category, X):
    artifacts = get_artifacts()
    labelencoder_1 = artifacts["labelencoder_1"]
    labelencoder_2 = artifacts["labelencoder_2"]
    labelencoder_3 = artifacts["labelencoder_3"]
    ct1 = artifacts["ct1"]
    ct2 = artifacts["ct2"]
    ct3 = artifacts["ct3"]

    w, h = 3, 1;
    new_col = [[0 for x in range(w)] for y in range(h)]
    num = 0

    for i in range(0, 1):
        new_col[i][0] = int(rating)
        new_col[i][1] = verified_purchase
        new_col[i][2] = product_category

    new_col = np.array(new_col)

    try:
        new_col[:, 0] = labelencoder_1.transform(new_col[:, 0])
        new_col[:, 1] = labelencoder_2.transform(new_col[:, 1])
        new_col[:, 2] = labelencoder_3.transform(new_col[:, 2])

        new_col = ct1.transform(new_col)
        try:
            new_col = new_col.toarray()
        except:
            pass
        new_col = new_col.astype(np.float64)

        new_col = ct2.transform(new_col)
        try:
            new_col = new_col.toarray()
        except:
            pass
        new_col = new_col.astype(np.float64)

        new_col = ct3.transform(new_col)
        try:
            new_col = new_col.toarray()
        except:
            pass
        new_col = new_col.astype(np.float64)
    except Exception:
        # Fallback for sklearn pickle compatibility issues (e.g. _infrequent_enabled).
        rating_classes = [str(x) for x in getattr(labelencoder_1, "classes_", [1, 2, 3, 4, 5])]
        verified_classes = [str(x) for x in getattr(labelencoder_2, "classes_", ["N", "Y"])]
        category_classes = [str(x) for x in getattr(labelencoder_3, "classes_", categories)]

        rating_value = str(int(rating))
        verified_value = str(verified_purchase)
        category_value = str(product_category)

        if rating_value not in rating_classes:
            raise ValueError("Unknown rating value '{}' for trained model.".format(rating_value))
        if verified_value not in verified_classes:
            raise ValueError("Unknown verified_purchase value '{}' for trained model.".format(verified_value))
        if category_value not in category_classes:
            raise ValueError("Unknown product_category value '{}' for trained model.".format(category_value))

        rating_vec = np.zeros(len(rating_classes), dtype=np.float64)
        verified_vec = np.zeros(len(verified_classes), dtype=np.float64)
        category_vec = np.zeros(len(category_classes), dtype=np.float64)

        rating_vec[rating_classes.index(rating_value)] = 1.0
        verified_vec[verified_classes.index(verified_value)] = 1.0
        category_vec[category_classes.index(category_value)] = 1.0

        # Match original ct pipeline final order: category one-hot, verified one-hot, rating one-hot.
        new_col = np.concatenate([category_vec, verified_vec, rating_vec]).reshape(1, -1)

    X= np.append(X, new_col, axis=1)
    return X

def POS_Tagging(sentence):
    ensure_nltk_resources()
    tagged_list = []
    tags = []
    count_verbs = 0
    count_nouns = 0
    text=nltk.word_tokenize(sentence)
    tagged_list = (nltk.pos_tag(text))

    tags = [x[1] for x in tagged_list]
    for each_item in tags:
        if each_item in ['VERB','VB','VBN','VBD','VBZ','VBG','VBP']:
            count_verbs+=1
        elif each_item in ['NOUN','NNP','NN','NUM','NNS','NP','NNPS']:
            count_nouns+=1
        else:
            continue
    if count_verbs > count_nouns:
        sentence = 'F'
    else:
        sentence = 'T'

    return sentence


def postag(sentence, X):
    w, h = 2, 1;
    pos_tag = [[0 for x in range(w)] for y in range(h)]
    num = 0

    sentence = POS_Tagging(sentence)

    if sentence=='T':
        pos_tag[0][0] = 1
        pos_tag[0][1] = 0
    else:
        pos_tag[0][0] = 0
        pos_tag[0][1] = 1

    X = np.append(X, pos_tag, axis=1)
    return X


def classify(X):
    bernoullinb = get_artifacts()["bernoullinb"]
    return bernoullinb.predict(X)


def classify_proba(X):
    bernoullinb = get_artifacts()["bernoullinb"]
    if hasattr(bernoullinb, "predict_proba"):
        return bernoullinb.predict_proba(X)
    return None

def get_result(statement, rating, verified_purchase, product_category):
    cleaned_statement = clean_review(statement)
    X = countvectorize([cleaned_statement])
    X = postag(statement, X)
    X = onehotencode(rating, verified_purchase, product_category, X)

    X = classify(X)
    return X


def predict_review(statement, rating, verified_purchase, product_category) -> Dict[str, Any]:
    valid = test_input(str(rating), str(verified_purchase), str(product_category), verbose=False)
    if not all(valid):
        errors = []
        if not valid[0]:
            errors.append("Rating must be between 1 and 5.")
        if not valid[1]:
            errors.append("Verified purchase must be Y or N.")
        if not valid[2]:
            errors.append("Product category is invalid.")
        return {"ok": False, "errors": errors}

    cleaned_statement = clean_review(statement)
    X = countvectorize([cleaned_statement])
    X = postag(statement, X)
    X = onehotencode(int(rating), verified_purchase, product_category, X)

    pred = int(classify(X)[0])
    proba = classify_proba(X)
    confidence = None
    if proba is not None:
        confidence = float(np.max(proba[0]))

    # Existing project mapping treats class 1 as True/Real.
    label = "REAL" if pred == 1 else "FAKE"
    return {
        "ok": True,
        "prediction": pred,
        "label": label,
        "confidence": confidence,
    }

def test_input(product_rating, verified_purchase, product_category, verbose=True) :
    x = True
    y = True
    z = True

    if product_rating != '1' and product_rating != '2' and product_rating != '3' and product_rating != '4' and product_rating != '5' :
        if verbose:
            print ("--------------------------------------------------------------------------------------.")
            print ("\nError : Product Rating must be Between 1 and 5 (inclusive).")
            print ("\nPlease Try Again.")

        x = False

    if verified_purchase != 'Y' and verified_purchase != 'N' :
        if verbose:
            print ("--------------------------------------------------------------------------------------.")
            print ("\nError : Verified Purchase must be either Y (Yes) or N (No).")
            print ("\nPlease Try Again.")

        y = False

    if product_category not in categories:
        if verbose:
            print ("--------------------------------------------------------------------------------------.")
            print ("\nError : Product Category must be among these choices : \n" + categories_str)
            print ("\nPlease Try Again.")

        z = False

    return [x, y, z]

if __name__ == '__main__':

    review_text = input("\nEnter your Review : ")

    product_rating = ""
    verified_purchase = ""
    product_category = ""

    input_ar = [False, False, False]

    while (True) :
        print("\n---------------------------------------------------------------------------------------\n")

        if not input_ar[0] :
            product_rating = input("\nEnter your Product Rating (On a scale of 1 to 5) : ")

        if not input_ar[1] :
            verified_purchase = input("\nEnter if it's a Verified Purchase (Y or N) : ")

        if not input_ar[2] :
            product_category = input("\nEnter your Product Category (" + categories_str + ") : ")

        input_ar = test_input(product_rating, verified_purchase, product_category)

        if input_ar == [True, True, True] :
            break

    result = predict_review(review_text, product_rating, verified_purchase, product_category)
    if not result["ok"]:
        print("\nInvalid input: {}".format("; ".join(result["errors"])))
    elif result["prediction"] == 1:
        print ("It is a True Review")

    else:
        print ("It is a False Review")
