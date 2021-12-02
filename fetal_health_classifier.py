import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest

dataset = pd.read_csv('./datasets/fetal_health.csv')

''' 

    Context

    Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals and is a key indicator of human progress.
    The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age, with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

    Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017). The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.

    In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality. The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.
    Data

    This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by three expert obstetritians into 3 classes:

        Normal = 0
        Suspect = 1
        Pathological = 2


'''
MODEL_COUNT = 0
BEST_MODEL = None
TOTAL_SCORE = 0
TOTAL_FEATURES = 1
BEST_MODEL_NAME = 'BEST_MODEL_FETAL_HEALTH'
# Preprocess dataset
dataset.dropna(inplace=True)
X = dataset.drop(columns=['fetal_health'])
y = dataset['fetal_health']

models_result_dataset = pd.DataFrame(columns=['F1', 'ACC', 'JACCARD', 'AVG_SCORE', 'TOTAL_SCORE'])

# params of models
tree_p = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)}
svc_p = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
knn_p = {'n_neighbors': range(1, 20, 1), 'weights': ['uniform', 'distance'],
         'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

for index, num_features in enumerate(range(1, len(X.columns), 1)):
    # Remove unused columns, select K best and then normalize it
    X_ = SelectKBest(k=num_features).fit_transform(X, y)
    X_ = StandardScaler().fit_transform(X_)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_, y, shuffle=True, stratify=y, test_size=0.3, random_state=42)

    for index_, (estimator, param) in enumerate(
            [(DecisionTreeClassifier(), tree_p), (SVC(), svc_p), (KNeighborsClassifier(), knn_p)]):
        grid_search = GridSearchCV(estimator=estimator, param_grid=param)

        # Train model and find best model with num_features
        grid_search.fit(X_train, y_train)
        predictor_name = str(grid_search.best_estimator_) + "_" + str(num_features) + '_features'
        prediction = grid_search.best_estimator_.predict(X_)

        # Get model score
        model_score = grid_search.best_estimator_.score(X_test, y_test)
        f1 = f1_score(y, prediction, average='weighted')
        acc = accuracy_score(y, prediction)
        jacc = jaccard_score(y, prediction, average='weighted')
        total_score = (model_score + 2 * f1 + acc + 2 * jacc) / 6

        # Store information inside dataframe
        models_result_dataset.loc[predictor_name, 'AVG_SCORE'] = model_score
        models_result_dataset.loc[predictor_name, 'F1'] = f1
        models_result_dataset.loc[predictor_name, 'ACC'] = acc
        models_result_dataset.loc[predictor_name, 'JACCARD'] = jacc
        models_result_dataset.loc[predictor_name, 'TOTAL_SCORE'] = total_score

        print(f'Model {MODEL_COUNT + 1} got total-score of {total_score}\t {predictor_name}')
        if total_score > TOTAL_SCORE:
            TOTAL_SCORE = total_score
            BEST_MODEL = grid_search.best_estimator_
            TOTAL_FEATURES = num_features
        MODEL_COUNT += 1

# Sort descending by best models
models_result_dataset.sort_values(by=['TOTAL_SCORE'], ascending=False, inplace=True)

with open(file=f'./models/fetal_health/{BEST_MODEL_NAME}.pickle', mode='wb') as model:
    pickle.dump(BEST_MODEL, model)
    X_ = SelectKBest(k=TOTAL_FEATURES).fit_transform(X, y)
    X_ = StandardScaler().fit_transform(X_)
    print('\nFINAL CONFUSION MATRIX:\n', confusion_matrix(y, BEST_MODEL.predict(X_, y)))

models_result_dataset.to_csv(path_or_buf='./models/fetal_health/fetal_health.csv')
