import pandas as pd
df = pd.read_csv('train.csv')
df.drop(['relation', 'people_main', 'city', 
'occupation_name', 'langs', 'bdate', 'graduation', 
'last_seen', 'education_status', 'followers_count', 'life_main','id' ], axis = 1, inplace = True)

male = 0 
female = 0

def male_female_undegraduate(row):
    global male, female
    if row['sex'] == 2 and row['result'] == 1:
        male += 1
    if row['sex'] == 1 and row['result'] == 1:
        female += 1
    return False

df.apply(male_female_undegraduate, axis = 1)
s = pd.Series(data= [male, female],
index = ['девушки', 'мужчины'])
s.plot(kind = 'barh')
plt.show()

def fill_education(education_form):
    if education_form == 'Distance Learning':
        return 1
    return 2
df['education_form'] = df['education_form'].apply(fill_education)

def fill_occp(occupation_type):
    if occupation_type == 'work':
        return 1
    return 2
df['occupation_type'] = df['occupation_type'].apply(fill_occp)

def fill_crrst(career_start):
    if career_start == 'False':
        return 1
    return 2
df['career_start'] = df['career_start'].apply(fill_crrst)

def fill_crrend(career_end):
    if career_end == 'False':
        return 1
    return 2
df['career_end'] = df['career_end'].apply(fill_crrend)

def fill_mb(has_mobile):
    if has_mobile == 1.0:
        return 1
    return 2
df['has_mobile'] = df['has_mobile'].apply(fill_mb)
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('result', axis = 1)
Y = df['result']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print('Процент правильно предсказанных исходов', round(accuracy_score(Y_test, Y_pred), 2) * 100)
print('Confusion_matrix')
print(confusion_matrix(Y_test, Y_pred))