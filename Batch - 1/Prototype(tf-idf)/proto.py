import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

csv_file = 'bbc_body_categories_old.csv'

input_data = pd.read_csv(csv_file, header=0)

X = input_data['body'].values.tolist()
y = input_data['categories'].values.tolist()
le = LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
                     ])

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

gs_clf = gs_clf.fit(X_train, y_train)
text_clf = text_clf.fit(X_train, y_train)

gs_clf.best_score_
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

ans = gs_clf.predict(X_test)
for a, b in zip(ans, y_test):
    print("%d => %d" % (a, b))

test_text = ["""Donald Glover has told the BBC he has been inspired to write new music by living in the UK.

“You guys have been very instrumental in my music. London has been very inspirational," he told a BBC Radio 1’s Clara Amfo.

He also addressed what people should call him now that Childish Gambino is coming to an end. “Yeah but it’s not yet”, he said, “you still have a lot more time of calling me that if you want”.

Glover has a pretty hectic schedule at the minute.

As well as writing new music, he has a role in the new Star Wars film as Lando Calrissian... not to mention his part in the new remake of The Lion King.

Don't forget to have a rest, Donald."""]

ans = gs_clf.predict(test_text)
print(le.inverse_transform(ans))
