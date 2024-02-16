from scipy import spatial, stats
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.stats import shapiro, normaltest, anderson
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as stats
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import missingno as msno
import seaborn as sns
import re as re1
import ast
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn import metrics
import time
import pickle
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import radians, cos, sin, asin, sqrt
from time import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pywaffle import Waffle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
from math import radians, cos, sin, asin, sqrt
from time import time
def regressor_compar(X_train, X_test, y_train, y_test, models, grids, foldername):
    '''Modeling of our algorithms with performance summary table.
       Save templates for easy use…'''

    print('Comparative table of models…')

    cols = ['extratree', 'dummy', 'lr', 'ridge', 'lasso',
            'elastic', 'knn', 'svr', 'rfr', 'gradboost']
    index = ['Standard Time', 'GridSearch Time', 'RandomSearch Time',
             'Standard R²', 'GridSearch R²', 'RandomSearch R²',
             'Standard RMSE', 'GridSearch RMSE', 'RandomSearch RMSE']
    results = pd.DataFrame(index=index, columns=cols)

    col = 0
    for model, grid in zip(models, grids):
        start_time = time.time()

        # Standard
        model.fit(X_train, y_train)
        standard_time = time.time() - start_time
        results.iloc[0, col] = standard_time
        results.iloc[3, col] = r2_score(y_test, model.predict(X_test))
        results.iloc[6, col] = np.sqrt(
            mean_squared_error(y_test, model.predict(X_test)))

        # GridSearch
        grid_search = GridSearchCV(
            estimator=model, param_grid=grid, n_jobs=-1, cv=5)
        grid_search.fit(X_train, y_train)
        grid_search_time = time.time() - start_time
        results.iloc[1, col] = grid_search_time
        results.iloc[4, col] = r2_score(y_test, grid_search.predict(X_test))
        results.iloc[7, col] = np.sqrt(
            mean_squared_error(y_test, grid_search.predict(X_test)))

        # RandomizedSearch
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=grid, n_iter=10, cv=5, n_jobs=-1)
        random_search.fit(X_train, y_train)
        random_search_time = time.time() - start_time
        results.iloc[2, col] = random_search_time
        results.iloc[5, col] = r2_score(y_test, random_search.predict(X_test))
        results.iloc[8, col] = np.sqrt(mean_squared_error(
            y_test, random_search.predict(X_test)))

        # Sauvegarde des modèles
        filename = f"{str(model).split('(')[0]}"
        with open(f"{foldername}{filename}_standard.pkl", 'wb') as file:
            pickle.dump(model, file)
        with open(f"{foldername}{filename}_gridsearch.pkl", 'wb') as file:
            pickle.dump(grid_search, file)
        with open(f"{foldername}{filename}_randomsearch.pkl", 'wb') as file:
            pickle.dump(random_search, file)

        col += 1

    return results


def data_preprocessing1(data, categorical_columns, numerical_columns, target, test_size):
    '''Standardize, encode, and split data'''

    X = data.copy()
    y = X.pop(target)

    # Standardiser les caractéristiques numériques
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encoder les caractéristiques catégorielles en utilisant OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[categorical_columns])

    # Fusionner les caractéristiques numériques standardisées avec les caractéristiques catégorielles encodées
    #X_encoded = np.concatenate((X[numerical_columns], X_categorical), axis=1)
    X_encoded = pd.merge(X[numerical_columns],
                         pd.DataFrame(
                             columns=encoder.get_feature_names_out(), data=X_categorical),
                         left_index=True, right_index=True)
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size)

    return X_encoded, X_train, X_test, y_train, y_test


def regression_visualizers(get_model, X_test, y_test, get_folder):
    '''Visualization by histogram, scatterplot, QQ-plot
        errors and actual values vs prediction.'''

    # Charger le modèle depuis le fichier spécifié
    model = pickle.load(open(get_folder + '/' + get_model + '.pkl', 'rb'))
    y_pred = model.predict(X_test)
    error = y_test - y_pred

    plt.figure(figsize=(20, 10))
    plt.subplot(221)
    sns.scatterplot(x=error, y=y_pred, hue=error, color='slateblue')
    plt.axhline(0, color='red', linestyle='dashed')
    plt.title("error vs prediction")
    plt.xlabel('')

    # Utiliser la palette dans sns.scatterplot
    plt.subplot(222)
    sns.scatterplot(x=y_test, y=y_pred, hue=y_test, color='slateblue')
    plt.title("real vs prediction")
    plt.xlabel('')

    plt.subplot(223)
    sns.distplot(error, kde=True, color='slateblue')
    plt.title("error distribution")
    plt.xlabel('')

    plt.subplot(224)
    stats.probplot(y_pred, plot=sns.mpl.pyplot)
    plt.xlabel('prediction')
    plt.ylabel('')

    plt.show()


def load_pickle(f_name):
    # If file of models exists, open and load in dict_model
    if os.path.exists(f_name):
        with open(f_name, "rb") as f:
            d_file = dill.load(f)
        print('--Pickle containing models already existing as ',
              f_name, ':\n', d_file.keys())
        print("Content loaded from '", f_name, "'.")
        return d_file
    # Else create an empty dictionary
    else:
        print('--No pickle yet as ', f_name)
        return {}


def extract_address_info(input_string):

    # Expression régulière pour extraire l'adresse, le code postal, la latitude, la longitude et l'état
    address_zip_regex = r'(\d+)\s+(.+)\n(.+),\s+WA\s+(\d{5})\n\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)'

    # Trouver les correspondances dans la chaîne
    matches = re1.search(address_zip_regex, input_string)

    if matches:
        street_number = matches.group(1)
        street_name = matches.group(2)
        city = matches.group(3)
        zip_code = matches.group(4)
        latitude = float(matches.group(5))
        longitude = float(matches.group(6))
        state = "WA"  # Ajoutez l'État ici
        # Retourner les informations sous forme de dictionnaire
        return {
            'StreetNumber': street_number,
            'StreetName': street_name,
            'City': city,
            'State': state,
            'ZipCode': zip_code,
            'Latitude': latitude,
            'Longitude': longitude
        }
    else:
        return None


def split_dates(dates):
    ls_date = []  # Initialisez ls_date avec une liste vide
    dates_str = str(dates).replace('nan', '')
    if ',' in dates_str:
        ls_date = [s.strip() for s in str(dates_str).split(',')]
    else:
        if len(dates_str) % 4 == 0:
            ls_date = [dates_str[4*(i):4*(i+1)]
                       for i in range(int(len(dates_str)/4))]
        else:
            print("ERROR: ", dates_str)
    return tuple(ls_date)
# Data preprocessing, Knn training, then predicting all-in-one
# inferring 'pnns1' (2_C) from .... + 'pnns2' (2_C)
# Works for both quantitative (knnregressor)
# and categorical (knnclassifier) target features


def knn_impute(df, var_model, var_target, enc_strat_cat='label',
               clip=None, plot=True):

    if df[var_target].isna().sum() == 0:
        print('ERROR: Nothing to impute (target column already filled)')
        return None, None
    else:
        if df[var_target].dtype == 'object':
            # knn classifier
            print('ooo----KNN CLASSIFICATION :', var_target)
            skf = StratifiedKFold(n_splits=5, shuffle=True)
            gsCV = GridSearchCV(KNeighborsClassifier(),
                                {'n_neighbors': [5, 7, 9, 11, 13]},
                                cv=skf, return_train_score=True,
                                scoring='f1_weighted')
            mod = 'class'
        elif df[var_target].dtype in ['float64', 'int64']:
            # knn regressor
            print('ooo----KNN REGRESSION :', var_target)
            kf = KFold(n_splits=5, shuffle=True)
            gsCV = GridSearchCV(KNeighborsRegressor(),
                                {'n_neighbors': [3, 5, 7, 9, 11, 13]},
                                cv=kf, return_train_score=True)
            mod = 'reg'
        else:
            print("ERROR: dtype of target feature unknown")
        # Data Preprocessing
        X, y = data_preprocessing(df.dropna(subset=var_model+[var_target]),
                                  var_model=var_model, var_target=var_target,
                                  enc_strat_cat=enc_strat_cat)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
        # Training KNN
        gsCV.fit(X_tr, y_tr)
        res = gsCV.cv_results_
        # Predicting test set with the model and clipping
        y_pr = gsCV.predict(X_te)
        try:
            if clip:
                y_pr = y_pr.clip(*clip)  # regressor option only
        except:
            print("ERROR: clip available for regressor option only")
        # Comparison with naive baselines
        if mod == 'class':
            naive_model_compare_acc_f1(
                X_tr, y_tr, X_te, y_te, y_pr, average='micro')
        elif mod == 'reg':
            naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr)
        else:
            print("ERROR: check type of target feature...")
        # Predicting using knn
        ind_to_impute = df.loc[df[var_target].isna()].index
        X_, y_ = data_preprocessing(df.loc[ind_to_impute], var_model=var_model,
                                    var_target=var_target,
                                    enc_strat_cat=enc_strat_cat)
        # Predicting with model
        y_pr_ = gsCV.predict(X_)
        # Plotting histogram of predicted values
        short_lab = True if mod == 'class' else False
        if plot:
            plot_hist_pred_val(y_te, y_pr, y_pr_, short_lab=short_lab)
        # returning indexes to impute and calculated values
        return ind_to_impute, y_pr_


################### End of KNN Imputation ########################

def handle_imputation_results(data, ind_to_impute, var_target, y_pr_):
    if ind_to_impute is not None:
        print(f'Number of values to impute for {var_target}:', len(
            ind_to_impute))
        data.loc[ind_to_impute, var_target] = y_pr_
    else:
        print(f'No values to impute for {var_target}')
# Plotting heatmap (2 options available, rectangle or triangle )


def plot_heatmap(corr, title, figsize=(8, 4), vmin=-1, vmax=1, center=0,
                 palette=sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False):

    fig, ax = plt.subplots(figsize=figsize)
    if shape == 'rect':
        mask = None
    elif shape == 'tri':

        mask = np.zeros_like(corr, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    palette = palette
    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10}, fmt=fmt,
                     square=False, linewidths=.5, linecolor='white',
                     cbar_kws={"shrink": .9, 'label': None}, robust=robust,
                     xticklabels=corr.columns, yticklabels=corr.index)
    ax.tick_params(labelsize=10, top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=25,
             ha="right", rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=12)


# Plotting explained variance ratio in scree plot

def scree_plot(col_names, exp_var_rat, ylim=(0, 0.4), figsize=(8, 3)):
    plt.bar(x=col_names, height=exp_var_rat, color='grey')
    ax1 = plt.gca()
    ax1.set(ylim=ylim)
    ax2 = ax1.twinx()
    ax2.plot(exp_var_rat.cumsum(), 'ro-')
    ax2.set(ylim=(0, 1.1))
    ax1.set_ylabel('explained var. rat.')
    ax2.set_ylabel('cumulative explained var. rat.')

    for i, p in enumerate(ax1.patches):
        ax1.text(p.get_width() / 5 + p.get_x(), p.get_height() + p.get_y() + 0.01,
                 '{:.0f}%'.format(exp_var_rat[i] * 100),
                 fontsize=8, color='k')

    plt.gcf().set_size_inches(figsize)
    plt.title('Scree plot', fontweight='bold')


def display_factorial_planes(X_projected,
                             x_y,
                             pca=None,
                             labels=None,
                             clusters=None,
                             alpha=1,
                             figsize=[10, 8],
                             marker="."):
    """
    Affiche la projection des individus

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize:
        figsize = (7, 6)

    # On gère les labels
    if labels is None:
        labels = []
    try:
        len(labels)
    except Exception as e:
        raise e

    # On vérifie la variable axis
    if not len(x_y) == 2:
        raise AttributeError("2 axes sont demandées")
    if max(x_y) >= X_.shape[1]:
        raise AttributeError("la variable axis n'est pas bonne")

    # on définit x et y
    x, y = x_y

    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters

    # Les points
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca:
        v1 = str(round(100*pca.explained_variance_ratio_[x])) + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y])) + " %"
    else:
        v1 = v2 = ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() * 1.1
    y_max = np.abs(X_[:, y]).max() * 1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom=-y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0, 0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels):
        # j'ai copié collé la fonction sans la lire
        for i, (_x, _y) in enumerate(X_[:, [x, y]]):
            plt.text(_x, _y+0.05, labels[i],
                     fontsize='14', ha='center', va='center')

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


def correlation_graph(pca,
                      x_y,
                      features):
    """Affiche le graphe des correlations

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y
    x, y = x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0, 0,
                 pca.components_[x, i],
                 pca.components_[y, i],
                 head_width=0.07,
                 head_length=0.07,
                 width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                 pca.components_[y, i] + 0.05,
                 features[i])

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(
        x+1, round(100*pca.explained_variance_ratio_[x], 1)))
    plt.ylabel('F{} ({}%)'.format(
        y+1, round(100*pca.explained_variance_ratio_[y], 1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)
# Plotting bar plots of the main categorical columns


def plot_barplots(df, cols, file_name=None, figsize=(12, 7), layout=(2, 3), save_enabled=False):

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols, 1):
        ax = fig.add_subplot(*layout, i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat > 15:
            ser[0:15].plot.bar(color='grey', ec='k', ax=ax)
        else:
            ser.plot.bar(color='grey', ec='k', ax=ax)
        ax.set_title(c[0:17]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s) > 7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_enabled:
        plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400)
    plt.show()
# Data Preprocessing for quantitative and categorical data with encoding options


def data_preprocessing(df, var_model, var_target, enc_strat_cat='label'):
    # Data Processing
    df_train = df[var_model+[var_target]].copy('deep')
    if df[var_model].isna().sum().sum() != 0:
        print("ERROR preprocessing: var_model columns should not contain nan !!!")
        return None, None
    else:
        cat_cols = df_train[var_model].select_dtypes('object').columns
        num_cols = df_train[var_model].select_dtypes(include=np.number).columns
        # # Encoding categorical values
        if enc_strat_cat == 'label':
            # --- OPTION 1: Label Encoding categorical values
            for c in cat_cols:
                df_train[c] = LabelEncoder().fit_transform(df_train[c].values)
        elif enc_strat_cat == 'hashing':
            # --- OPTION 2: Feature hashing of categorical values
            for c in cat_cols:
                df_train[c] = df_train[c].astype('str')
                n_feat = 5
                hasher = FeatureHasher(n_features=n_feat, input_type='string')
                f = hasher.transform(df_train[c])
                arr = pd.DataFrame(f.toarray(), index=df_train.index)
                df_train[[c+'_'+str(i+1)
                          for i in range(n_feat)]] = pd.DataFrame(arr)
                del df_train[c]
                cols = list(df_train.columns)
                cols.remove(var_target)
                df_train = df_train.reindex(columns=cols+[var_target])
        else:
            print("ERROR: Wrong value of enc_strat_cat")
            return None, None
        # # Standardizing quantitative values
        if len(list(num_cols)):
            df_train[num_cols] = \
                StandardScaler().fit_transform(df_train[num_cols].values)
        # Splitting in X and y, then in training and testing set
        X = df_train.iloc[:, :-1].values
        y = df_train.iloc[:, -1].values
        return X, y
################### KNN Imputation ########################


def naive_model_compare_r2(X_tr, y_tr, X_te, y_te, y_pr):
    # Model
    print('--- model: {:.3}'.format(metrics.r2_score(y_te, y_pr)))
    # normal random distribution
    y_pr_rand = np.random.normal(0, 1, y_pr.shape)
    print('--- normal random distribution: {:.3}'
          .format(metrics.r2_score(y_te, y_pr_rand)))
    # dummy regressors
    for s in ['mean', 'median']:
        dum = DummyRegressor(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy regressor (' + s + ') : r2_score={:.3}'
              .format(metrics.r2_score(y_te, y_pr_dum)))


def naive_model_compare_acc_f1(X_tr, y_tr, X_te, y_te, y_pr, average='weighted'):
    def f1_prec_recall(yte, ypr):
        prec = metrics.precision_score(yte, ypr, average=average)
        rec = metrics.recall_score(yte, ypr, average=average)
        f1 = metrics.f1_score(yte, ypr, average=average)
        return [f1, prec, rec]
    # Model
    print('--- model: f1={:.3}, precision={:.3}, recall={:.3}'
          .format(*f1_prec_recall(y_te, y_pr)))
    # Dummy classifier
    for s in ['stratified', 'most_frequent', 'uniform']:
        dum = DummyClassifier(strategy=s).fit(X_tr, y_tr)
        y_pr_dum = dum.predict(X_te)
        print('--- dummy class. (' + s
              + '): f1={:.3}, precision={:.3}, recall={:.3}'
              .format(*f1_prec_recall(y_te, y_pr_dum)))


def plot_hist_pred_val(y_te, y_pr, y_pr_, bins=150, xlim=(0, 20), short_lab=False):
    # Plotting dispersion of data to be imputed
    bins = plt.hist(y_te, alpha=0.5, color='b', bins=bins, density=True,
                    histtype='step', lw=3, label='y_te (real val. from test set)')[1]
    ax = plt.gca()
    ax.hist(y_pr, alpha=0.5, color='g', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr (pred. val. from test set)')
    ax.hist(y_pr_, alpha=0.5, color='r', bins=bins, density=True,
            histtype='step', lw=3, label='y_pr_ (pred. val. to be imputed)')
    ax.set(xlim=xlim)
    plt.xticks(rotation=45, ha='right')
    plt.draw()
    if short_lab:
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s) > 7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
    ax.legend(loc=1)
    plt.title("Frequency of values", fontweight='bold', fontsize=12)
    plt.gcf().set_size_inches(6, 2)
    plt.show()


def plot_histograms(df, cols, file_name=None, bins=30, figsize=(12, 7), skip_outliers=True, thresh=3, layout=(3, 3), save_enabled=False):

    fig = plt.figure(figsize=figsize)

    for i, c in enumerate(cols, 1):
        ax = fig.add_subplot(*layout, i)
        if skip_outliers:
            ser = df[c][np.abs(st.zscore(df[c])) < thresh]
        else:
            ser = df[c]
        ax.hist(ser,  bins=bins, color='grey')
        ax.set_title(c)
        ax.vlines(df[c].mean(), *ax.get_ylim(),  color='red', ls='-', lw=1.5)
        ax.vlines(df[c].median(), *ax.get_ylim(),
                  color='green', ls='-.', lw=1.5)
        ax.vlines(df[c].mode()[0], *ax.get_ylim(),
                  color='goldenrod', ls='--', lw=1.5)
        ax.legend(['mean', 'median', 'mode'])
        ax.title.set_fontweight('bold')
        # xmin, xmax = ax.get_xlim()
        # ax.set(xlim=(0, xmax/5))

    plt.tight_layout(w_pad=0.5, h_pad=0.65)

    if save_enabled:
        plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400)
    plt.show()


# plotting histograms, qq plots and printing the results of normality tests


def plot_hist_qqplot(data, name, save=False):
    fig, axs = plt.subplots(1, 2)
    # histogram
    axs[0].hist(data, histtype='stepfilled',
                ec='k', color='lightgrey', bins=25)
    # using statsmodels qqplot's module
    qqplot(data, line='r', **{'markersize': 5,
           'color': 'lightgrey'}, ax=axs[1])
    plt.gcf().set_size_inches(10, 2.5)
    fig.suptitle(name, fontweight='bold', size=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])

# Plotting bar plots of the main categorical columns


def plot_barplots(df, cols, file_name=None, figsize=(12, 7), layout=(2, 3), save_enabled=False):

    fig = plt.figure(figsize=figsize)
    for i, c in enumerate(cols, 1):
        ax = fig.add_subplot(*layout, i)
        ser = df[c].value_counts()
        n_cat = ser.shape[0]
        if n_cat > 15:
            ser[0:15].plot.bar(color='grey', ec='k', ax=ax)
        else:
            ser.plot.bar(color='grey', ec='k', ax=ax)
        ax.set_title(c[0:17]+f' ({n_cat})', fontweight='bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [s[0:7]+'.' if len(s) > 7 else s for s in labels]
        ax.axes.set_xticklabels(short_labels)
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_enabled:
        plt.savefig(os.getcwd()+'/FIG/'+file_name, dpi=400)
    plt.show()


def normality_tests(data, print_opt=False):
    res_df = pd.DataFrame([])
    data_notna = data[data.notna()]
    # Shapiro-Wilk - D'Agostino's K^2
    for f_name, func in zip(['Shapiro-Wilk', "D'Agostino K^2"], [shapiro, normaltest]):
        stat, p = func(data_notna)
        res_df.loc[f_name, 'stat'] = stat
        res_df.loc[f_name, 'p_value'] = p
        if print_opt:
            print('---'+f_name)
        if print_opt:
            print('stat=%.3f, p=%.3f' % (stat, p))
        res_df.loc[f_name, 'res'] = [p > 0.05]
        if p > 0.05:
            if print_opt:
                print('Probably Gaussian')
        else:
            if print_opt:
                print('Probably not Gaussian')
    # Anderson-Darling
    result = anderson(data_notna)
    if print_opt:
        print('---'+'Anderson-Darling')
    res_df.loc['Anderson-Darling', 'stat'] = result.statistic
    if print_opt:
        print('stat=%.3f' % (result.statistic))
    res_and = [(int(result.significance_level[i]), result.statistic < res)
               for i, res in enumerate(result.critical_values)]
    res_df.loc['Anderson-Darling', 'res'] = str(res_and)
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            if print_opt:
                print('Probably Gaussian at the %.1f%% level' % (sl))
        else:
            if print_opt:
                print('Probably not Gaussian at the %.1f%% level' % (sl))
    return res_df


def correlation_tests(data1, data2, print_opt=False):
    res_df = pd.DataFrame([])
    # data1_notna = data1[data1.notna()]
    # Pearson, Spearman, Kendall
    for f_name, func in zip(['Pearson', 'Spearman', 'Kendall'], [pearsonr, spearmanr, kendalltau]):
        stat, p = func(data1, data2)
        res_df.loc[f_name, 'stat'] = stat
        res_df.loc[f_name, 'p_value'] = p
        if print_opt:
            print('---'+f_name)
        if print_opt:
            print('stat=%.3f, p=%.3f' % (stat, p))
        if print_opt:
            print('Probably independent') if p > 0.05 else print(
                'Probably dependent')
    return res_df

# Data Preprocessing for quantitative and categorical data with encoding options


def data_preprocessing(df, var_model, var_target, enc_strat_cat='label'):
    # Data Processing
    df_train = df[var_model+[var_target]].copy('deep')
    if df[var_model].isna().sum().sum() != 0:
        print("ERROR preprocessing: var_model columns should not contain nan !!!")
        return None, None
    else:
        cat_cols = df_train[var_model].select_dtypes('object').columns
        num_cols = df_train[var_model].select_dtypes(include=np.number).columns
        # # Encoding categorical values
        if enc_strat_cat == 'label':
            # --- OPTION 1: Label Encoding categorical values
            for c in cat_cols:
                df_train[c] = LabelEncoder().fit_transform(df_train[c].values)
        elif enc_strat_cat == 'hashing':
            # --- OPTION 2: Feature hashing of categorical values
            for c in cat_cols:
                df_train[c] = df_train[c].astype('str')
                n_feat = 5
                hasher = FeatureHasher(n_features=n_feat, input_type='string')
                f = hasher.transform(df_train[c])
                arr = pd.DataFrame(f.toarray(), index=df_train.index)
                df_train[[c+'_'+str(i+1)
                          for i in range(n_feat)]] = pd.DataFrame(arr)
                del df_train[c]
                cols = list(df_train.columns)
                cols.remove(var_target)
                df_train = df_train.reindex(columns=cols+[var_target])
        else:
            print("ERROR: Wrong value of enc_strat_cat")
            return None, None
        # # Standardizing quantitative values
        if len(list(num_cols)):
            df_train[num_cols] = \
                StandardScaler().fit_transform(df_train[num_cols].values)
        # Splitting in X and y, then in training and testing set
        X = df_train.iloc[:, :-1].values
        y = df_train.iloc[:, -1].values
        return X, y


def data_preprocessing1(data, categorical_columns, numerical_columns, target, test_size):
    '''Standardize, encode, and split data'''

    X = data.copy()
    y = X[target]
    print(f'X : ', X.shape)
    # Standardiser les caractéristiques numériques
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    X_categorical = pd.get_dummies(X[categorical_columns])
    # Encoder les caractéristiques catégorielles en utilisant OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X_categorical)

    # Fusionner les caractéristiques numériques standardisées avec les caractéristiques catégorielles encodées
    #X_encoded = np.concatenate((X[numerical_columns], X_categorical), axis=1)
    X_encoded = pd.merge(X[numerical_columns],
                         pd.DataFrame(
                             columns=encoder.get_feature_names_out(), data=X_categorical),
                         left_index=True, right_index=True)
    #X_encoded = pd.concat([X[numerical_columns], X_categorical], axis=1)

    print(f'X_encoded : ', X_encoded.shape)
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def evaluate_model_performance(model, X_test, y_test, model_name):
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)

    print(f'Métrique de performance pour le modèle {model_name} :')
    # Calcul des métriques d'évaluation
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'R²: {r2}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')

    # Graphique de Régression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
             linestyle='--', color='red', linewidth=2)
    plt.xlabel('Valeurs Réelles')
    plt.ylabel('Prédictions')
    plt.title(f'Graphique de Régression {model_name} ')
    plt.show()

    # Graphique de Distribution d'Erreurs
    plt.figure(figsize=(8, 6))
    sns.distplot(y_test - y_pred, bins=50, color='blue')
    plt.xlabel('Erreurs')
    plt.ylabel('Fréquence')
    plt.title(f'Graphique de Distribution d\'Erreurs {model_name} ')
    plt.show()

    # QQ Plot
    plt.figure(figsize=(8, 6))
    stats.probplot(y_test - y_pred, dist="norm", plot=plt)
    plt.title(f'QQ Plot {model_name} ')
    plt.show()

    # Graphique d'Importance des Caractéristiques (pour les modèles avec feature_importances_ attribut)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(
            model.feature_importances_, index=X_test.columns)
        # Sélectionner les 20 principales caractéristiques
        top_features = feature_importances.nlargest(20)
        top_features.plot(kind='barh')

        plt.xlabel('Importance')
        plt.title(f'Top 20 Features of {model_name}  ')
        plt.show()


def data_preprocessing2(data, categorical_columns, numerical_columns, target, test_size):
    '''Standardize, encode, and split data'''

    X = data.copy()
    y = X[target]

    # Standardiser les caractéristiques numériques
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encoder les caractéristiques catégorielles en utilisant OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[categorical_columns])

    # Créer un DataFrame à partir des caractéristiques catégorielles encodées
    encoded_categorical_df = pd.DataFrame(
        columns=encoder.get_feature_names_out(), data=X_categorical)

    # Fusionner les caractéristiques numériques standardisées avec les caractéristiques catégorielles encodées
    X_encoded = pd.concat([X[numerical_columns].reset_index(
        drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def calculate_mean_distance_for_outliers_drop(df, prop_Q_cols, k=6, quantile=0.99):
    test_kdtree = df[prop_Q_cols].copy()
    scaler = StandardScaler().fit(test_kdtree)
    scaled_data = scaler.transform(test_kdtree)
    scaled_tree = spatial.KDTree(scaled_data)
    neighbours_scaled = scaled_tree.query(scaled_data, k=6)
    dist_scaled = pd.DataFrame(neighbours_scaled[0])
    dist_scaled = dist_scaled.drop(columns=0)
    dist_scaled['mean'] = dist_scaled.mean(axis=1)
    mean_description = dist_scaled['mean'].describe()
    quantile_99 = dist_scaled['mean'].quantile(0.99)
    filtered_dist_scaled = dist_scaled[dist_scaled['mean'] < quantile_99]
    count_below_quantile = dist_scaled.shape[0] - filtered_dist_scaled.count()

    df['mean'] = dist_scaled['mean'].values
    dist_scaled = dist_scaled.drop(columns='mean')
    df = df.drop(columns='mean')

    return df, mean_description, quantile_99, count_below_quantile


def load_and_evaluate_models(folder_path, X_test, y_test, evaluate_function):
    """
    Charge les modèles à partir des fichiers pickle dans le dossier spécifié,
    puis évalue la performance de chaque modèle en utilisant la fonction d'évaluation donnée.

    Args:
    folder_path (str): Chemin du dossier contenant les fichiers de modèle pickle.
    X_test (array-like): Données de test.
    y_test (array-like): Labels de test.
    evaluate_function (function): Fonction pour évaluer la performance du modèle.
    """
    loaded_models = []
    # Parcourez tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        # Vérifiez si le fichier a l'extension .pkl
        if filename.endswith('.pkl'):
            # Chemin complet du fichier
            file_path = os.path.join(folder_path, filename)
            # Chargez le modèle à partir du fichier pickle
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            # Ajoutez le modèle à la liste des modèles chargés
            loaded_models.append(model)

            # Appelez la fonction pour évaluer la performance du modèle
            evaluate_function(model, X_test, y_test, filename)

    return loaded_models


def remove_outliers_with_kdtree(data, k=6, threshold_multiplier=1.5, numerical_columns=None, categorical_columns=None):
    # Standardiser les données
    scaler = StandardScaler()
    X = data.copy()

    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    # Encoder les caractéristiques catégorielles en utilisant OneHotEncoder
    encoder = OneHotEncoder(sparse=False)
    X_categorical = encoder.fit_transform(X[categorical_columns])

    # Créer un DataFrame à partir des caractéristiques catégorielles encodées
    encoded_categorical_df = pd.DataFrame(
        columns=encoder.get_feature_names_out(), data=X_categorical)

    # Fusionner les caractéristiques numériques standardisées avec les caractéristiques catégorielles encodées
    X_encoded = pd.concat([X[numerical_columns].reset_index(
        drop=True), encoded_categorical_df.reset_index(drop=True)], axis=1)

    scaled_tree = spatial.KDTree(X_encoded)
    neighbours_scaled = scaled_tree.query(X_encoded, k=6)
    dist_scaled = pd.DataFrame(neighbours_scaled[0])
    dist_scaled = dist_scaled.drop(columns=0)
    dist_scaled['mean'] = dist_scaled.mean(axis=1)
    dist_scaled['mean'].describe()
    data['mean'] = dist_scaled['mean'].values
    data = data.loc[data['mean'] < data['mean'].quantile(0.99)]
    return data
from sklearn.metrics import silhouette_score, davies_bouldin_score
from math import pi

def rmf_plot(data, label):
        # Frequency V Monetary
    fig, axe = plt.subplots(1, 3, figsize = (20, 5) )
    sns.set_context(context = 'notebook', font_scale=1)
    sns.scatterplot(
        x = 'frequency', 
        y = 'Monetary', 
        hue = label,
        data = data,
        palette='Set1', ax=axe[0])
    plt.title('Frequency V Monetary')
    plt.ylabel('Monetary')
    plt.xlabel('Frequency')
    # Monetary V Recency
    sns.set_context(context = 'notebook', font_scale=1)
    sns.scatterplot(
        x = 'recency', 
        y = 'total_payment_log', 
        hue = label,
        data = data,
        palette='Set1', ax=axe[1])
    plt.title('Recency V Monetary')
    plt.ylabel('Monetary')
    plt.xlabel('Recency')
    # Frequency V Recency
    sns.set_context(context = 'notebook', font_scale=1)
    sns.scatterplot(
        x = 'frequency', 
        y = 'recency', 
        hue = label,
        data = data,
        palette='Set1', ax=axe[2])
    plt.title('Frequency V Recency')
    plt.ylabel('Recency')
    plt.xlabel('Frequency')
    plt.show()


 
def radar_plot(data, label):
    # Set data
    df = data.groupby([label], as_index=False).agg({
                'recency': 'mean',
                'frequency': 'mean',
                'total_payment_log': 'mean',
                'freight_value': 'mean',
                'review_score': 'mean',
                'order_line_cube_in_ltr_log': 'mean',
                'avg_comment': 'mean',
            })
    
    # ------- PART 1: Create background
    
    # number of variable
    categories=list(df)[1:]
    N = len(categories)
    colorp = ['#4d86a5', '#cf0bf1', '#12e2f1', '#3e517a', '#98da1f', '#fc9f5b', '#d60b2d', '#9cc76d', '#2dffdf', '#c3c4e9']
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(10, 10))
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2,0.4,0.6,0.8, 1], ['0.2','0.4','0.6','0.8', '1'], color="grey", size=7)
    plt.ylim(0,1)
    

    # ------- PART 2: Add plots
    
    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable
    
    for x in range(len(df)):
        values=df.loc[x].drop(label).values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=f"Cluster : {x}", color=colorp[x])
        ax.fill(angles, values, colorp[x], alpha=0.2)
    
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Radar map of features and cluster")
    # Show the graph
    plt.show()




def plothist(data, huelabel):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))	
    sns.kdeplot(data=data, x="recency",  ax=axs[0, 0], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="frequency",  ax=axs[0, 1], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="total_payment",  ax=axs[0, 2], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="freight_value",  ax=axs[0, 3], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="review_score",  ax=axs[1, 0], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="order_line_cube_in_ltr",  ax=axs[1, 1], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="avg_comment",  ax=axs[1, 2], hue=huelabel, palette='Set2')
def plot_radars(data, group):

    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data), 
                        index=data.index,
                        columns=data.columns).reset_index()
    
    fig = go.Figure()

    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group]==k].iloc[:,1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster '+str(k)
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
        showlegend=True,
        title={
            'text': "Comparaison des moyennes par variable des clusters",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=18)

    fig.show()   
     
def haversine_distance(lat1, lng1, lat2, lng2, degrees=True):
    """The haversine formula makes it possible to determine the distance of the great circle 
        between two points of a sphere, from their longitudes and latitudes.

    Parameters
    ----------
    lat1, lat2 : float
        Latitudes of the 2 coordinate points to compare. 
    lng1, lng2 : float
        Longitudes of the 2 coordinate points to compare.
    degrees : boolean
        If True, converts radians to degrees.
    """
    # Radius of the earth in miles
    r = 3956 
    
    if degrees:
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
    
    # Haversine formula
    dlng = lng2 - lng1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
    d = 2 * r * asin(sqrt(a))  

    return d
def rfm_iso_scatter(dat, x, y, z):

    '''
    Function to generate a 3D scatter plot

    Input:
    - dat - dataframe
    - x, y, z - 3 features for the three coordinates

    Output:
    - None (interactive 3D scatter plot)
    '''
    x = dat[x]
    y = dat[dat[y]<5][y]
    z = dat[dat[z]<4000][z]


    fig = go.Figure(data=[go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=1,
            color=y,               
            colorscale='thermal',
            opacity=0.8
        )
    )])

    fig.update_layout(scene = dict(
        xaxis_title='Recency',
        yaxis_title='Frequency',
        zaxis_title='Monetary'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()
    
def plothist(data, huelabel):
    fig, axs = plt.subplots(2, 5, figsize=(20, 20))	
    sns.kdeplot(data=data, x="recency",  ax=axs[0, 0], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="frequency",  ax=axs[0, 1], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="total_payment",  ax=axs[0, 2], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="freight_value",  ax=axs[0, 3], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="review_score",  ax=axs[0, 4], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="order_line_cube_in_ltr",  ax=axs[1, 0], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="avg_comment",  ax=axs[1, 1], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="geolocation_lat",  ax=axs[1, 2], hue=huelabel, palette='Set2')
    sns.kdeplot(data=data, x="geolocation_zip_code_prefix",  ax=axs[1, 3], hue=huelabel, palette='Set2')   
    sns.kdeplot(data=data, x="geolocation_lng",  ax=axs[1, 4], hue=huelabel, palette='Set2')   
     

def plot_waffle_chart(dat, metric, agg, title_txt, group='sub_segment'):
        '''
        Funtion to create a waffle chart. The visualization shows how the
        customer sub-segments are distributed according defined metrics.

        Input:
        - dat - dataframe
        - metric - feature/ kpi metric to visualize
        - agg - method to aggregate
        - title_txt - text to display as chart title

        Outout:
        - waffle chart
        '''
        data_revenue = dict(round(
            dat.groupby(group).agg({metric: agg}))[metric])

        plt.figure(
            FigureClass=Waffle,
            rows=5,
            columns=10,
            values=data_revenue,
            labels=[f"{k, v}" for k, v in data_revenue.items()],
            legend={'loc': 'lower left', 'bbox_to_anchor': (1, 0)},
            figsize=(8, 5)
        )

        plt.title(title_txt)    


def plot_map(
        df,
        title,
        lower_bound,
        upper_bound,
        metric,
        maker_size=3,
        sub_segment_y_n=False
    ):
        '''
        Funtion for geographic data visualization of demographic metrics.

        Input:
        - df - dataframe with target feature
            (metric; color-code field needed if sub_segment to visualize
        - title - text to display as chart title
        - lower_bound - lower threshold for color scale
        - upper_bound - upper threshold for color scale
        - metric - feature/ kpi metric to visualize
        - sub_sement_y_n - boolean,
            if "True": sub-segment will be visualized with color-code,
            if no, color acording value
        - marker_size - size of the marker

        Outout:
        - geographic data visualization

        '''

        if sub_segment_y_n is True:
            dict_marker = dict(
                size=maker_size,
                color=df.color,
            )
        else:
            dict_marker = dict(
                size=maker_size,
                color=df[metric],
                showscale=True,
                colorscale=[[0, 'blue'],
                            [1, 'red']],
                cmin=lower_bound,
                cmax=upper_bound
            )

        data_geo = [go.Scattermapbox(
            lon=df['geolocation_lng'],
            lat=df['geolocation_lat'],
            marker=dict_marker
        )]

        layout = dict(
            title=title,
            showlegend=False,
            mapbox=dict(
                accesstoken='pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
                center=dict(lat=-23.5, lon=-46.6),
                bearing=10,
                pitch=0,
                zoom=2,
            )
        )
        fig = dict(data=data_geo, layout=layout)
        iplot(fig, validate=False)    
def create_colored_map(df, output_file='map.html'):
    """
    Crée une carte avec des marqueurs colorés en fonction des segments RFM.

    Args:
        df (DataFrame): Le DataFrame contenant les informations (latitude, longitude, segment).
        output_file (str): Le nom du fichier de sortie pour la carte (par défaut 'map.html').

    Returns:
        None
    """
    # Créer une carte centrée sur le Brésil
    brazil_map = folium.Map(location=[-14.235, -51.9253], zoom_start=4)

    segment_colors = {
        'Promising': 'red', # inactive
        'New Customers': 'orange', # active
        'Cannot Lose Them': 'yellow', # hot
        'Lost customers': 'green', # cold
        'Hibernating customers': 'blue',
        'About To Sleep': 'purple',
        'Champions': 'pink',
        'At Risk': 'brown',
        'Loyal': 'gray',
        'Potential Loyalist': 'black'
    }

    # Créer un nuage de points en fonction des coordonnées (latitude, longitude) et colorer en fonction des segments
    for index, row in df.iterrows():
        lat, lon, segment = row['geolocation_lat'], row['geolocation_lng'], row['Segments_RFM']
        
        # Vérifier si le segment est présent dans la palette de couleurs
        color = segment_colors.get(segment, 'gray')  # Gris par défaut pour les segments non définis
        
        # Ajouter le point à la carte
        folium.CircleMarker(
            location=[lat, lon],
            radius=1,  # Ajustez la taille des points selon vos préférences
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
        ).add_to(brazil_map)

    # Enregistrer la carte en HTML
    brazil_map.save(output_file)
    print(f"La carte a été enregistrée dans '{output_file}'.")




def make_dataset(dpath="datas/", initial=False, period=2):
    """Cleaning and feature engineering on complete Olist data 
        for preparation of unsupervised classification (K-Means).

    Parameters
    ----------
    dpath : str
        Path to the directory containing the data.
    initial : boolean
        Defines whether the created dataset is the initial dataset.
    period : int
        Increment period in months after initial dataset.
    """
    start_time = time()
    print("Création du dataset en cours ...")
    
    # Root path
    root_path = dpath
    
    # Load datasets
    customers = pd.read_csv(root_path + "olist_customers_dataset.csv")
    geolocation = pd.read_csv(root_path + "olist_geolocation_dataset.csv")
    orders = pd.read_csv(root_path + "olist_orders_dataset.csv")
    order_items = pd.read_csv(root_path + "olist_order_items_dataset.csv")
    order_payments = pd.read_csv(root_path + "olist_order_payments_dataset.csv")
    order_reviews = pd.read_csv(root_path + "olist_order_reviews_dataset.csv")
    products = pd.read_csv(root_path + "olist_products_dataset.csv")
    categories_en = pd.read_csv(root_path + "product_category_name_translation.csv")
    
    # Group location 
    geolocation = geolocation.groupby(["geolocation_state"]).agg({
            "geolocation_lat": "mean",
            "geolocation_lng": "mean"})
    
    # Merge datasets
    # Orders
    orders.drop(["order_approved_at",
                 "order_delivered_carrier_date", 
                 "order_estimated_delivery_date"],
                axis=1, inplace=True)

    order_items.drop(["seller_id",
                      "shipping_limit_date"],
                     axis=1, inplace=True)
    order_items = pd.merge(order_items, orders,
                           how="left",
                           on="order_id")
    
    datetime_cols = ["order_purchase_timestamp", 
                     "order_delivered_customer_date"]
    for col in datetime_cols:
        order_items[col] = order_items[col].astype('datetime64[ns]')
        
    # order Month
    order_items["sale_month"] = order_items['order_purchase_timestamp'].dt.month
    
    # Select orders on period
    start=order_items["order_purchase_timestamp"].min()
    if(initial == True):
        period = 12
    else:
        period = 12+period
    stop=start + pd.DateOffset(months=period)
        
    order_items = order_items[(order_items["order_purchase_timestamp"]>=start)
                              & (order_items["order_purchase_timestamp"]<stop)]
    
    # List of orders on period
    period_orders = order_items.order_id.unique()
    
    # Calculate other features on period
    order_payments = order_payments[order_payments["order_id"].isin(period_orders)]
    order_items = pd.merge(order_items, 
                           order_payments.groupby(by="order_id").agg(
                               {"payment_sequential": 'count',
                                "payment_installments": 'sum'}),
                           how="left",
                           on="order_id")
    order_items = order_items.rename(columns={
        "payment_sequential": "nb_payment_sequential",
        "payment_installments": "sum_payment_installments"})
    
    order_reviews = order_reviews[order_reviews["order_id"].isin(period_orders)]
    order_items = pd.merge(order_items,
                           order_reviews.groupby("order_id").agg({
                               "review_score": "mean"}),
                           how="left",
                           on="order_id")
    
    # Delivery time
    order_items["delivery_delta_days"] = (order_items.order_delivered_customer_date
                                          - order_items.order_purchase_timestamp)\
                                         .dt.round('1d').dt.days
    order_items.drop("order_delivered_customer_date", axis=1, inplace=True)
    
    # Products
    products = pd.merge(products, categories_en,
                    how="left",
                    on="product_category_name")

    del_features_list = ["product_category_name", "product_weight_g",
                         "product_length_cm", "product_height_cm",
                         "product_width_cm", "product_name_lenght", 
                         "product_description_lenght", "product_photos_qty"]
    products.drop(del_features_list, axis=1, inplace=True)
    products = products.rename(columns={"product_category_name_english":
                                        "product_category_name"})
        
    products['product_category'] = np.where((products['product_category_name'].str.contains("fashio|luggage")==True),
                                    'fashion_clothing_accessories',
                            np.where((products['product_category_name'].str.contains("health|beauty|perfum")==True),
                                     'health_beauty',
                            np.where((products['product_category_name'].str.contains("toy|baby|diaper")==True),
                                     'toys_baby',
                            np.where((products['product_category_name'].str.contains("book|cd|dvd|media")==True),
                                     'books_cds_media',
                            np.where((products['product_category_name'].str.contains("grocer|food|drink")==True), 
                                     'groceries_food_drink',
                            np.where((products['product_category_name'].str.contains("phon|compu|tablet|electro|consol")==True), 
                                     'technology',
                            np.where((products['product_category_name'].str.contains("home|furnitur|garden|bath|house|applianc")==True), 
                                     'home_furniture',
                            np.where((products['product_category_name'].str.contains("flow|gift|stuff")==True),
                                     'flowers_gifts',
                            np.where((products['product_category_name'].str.contains("sport")==True),
                                     'sport',
                                     'other')))))))))
    products.drop("product_category_name", axis=1, inplace=True)

    order_items = pd.merge(order_items, products, 
                           how="left",
                           on="product_id")
    
    # Encode categories column
    order_items = pd.get_dummies(order_items, columns=["product_category"], prefix="", prefix_sep="")
    
    # Customers
    order_items = pd.merge(order_items, customers[["customer_id",
                                                   "customer_unique_id",
                                                   "customer_state"]],
                           on="customer_id",
                           how="left")
    
    # Group datas by unique customers
    data = order_items.groupby(["customer_unique_id"]).agg(
        nb_orders=pd.NamedAgg(column="order_id", aggfunc="nunique"),
        total_items=pd.NamedAgg(column="order_item_id", aggfunc="count"),
        total_spend=pd.NamedAgg(column="price", aggfunc="sum"),
        total_freight=pd.NamedAgg(column="freight_value", aggfunc="sum"),
        mean_payment_sequential=pd.NamedAgg(column="nb_payment_sequential", aggfunc="mean"),
        mean_payment_installments=pd.NamedAgg(column="sum_payment_installments", aggfunc="mean"),
        mean_review_score=pd.NamedAgg(column="review_score", aggfunc="mean"),
        mean_delivery_days=pd.NamedAgg(column="delivery_delta_days", aggfunc="mean"),
        books_cds_media=pd.NamedAgg(column="books_cds_media", aggfunc="sum"),
        fashion_clothing_accessories=pd.NamedAgg(column="fashion_clothing_accessories", aggfunc="sum"),
        flowers_gifts=pd.NamedAgg(column="flowers_gifts", aggfunc="sum"),
        groceries_food_drink=pd.NamedAgg(column="groceries_food_drink", aggfunc="sum"),
        health_beauty=pd.NamedAgg(column="health_beauty", aggfunc="sum"),
        home_furniture=pd.NamedAgg(column="home_furniture", aggfunc="sum"),
        other=pd.NamedAgg(column="other", aggfunc="sum"),
        sport=pd.NamedAgg(column="sport", aggfunc="sum"),
        technology=pd.NamedAgg(column="technology", aggfunc="sum"),
        toys_baby=pd.NamedAgg(column="toys_baby", aggfunc="sum"),
        customer_state=pd.NamedAgg(column="customer_state", aggfunc="max"),
        first_order=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="min"),
        last_order=pd.NamedAgg(column="order_purchase_timestamp", aggfunc="max"),
        favorite_sale_month=pd.NamedAgg(column="sale_month", 
                                        aggfunc=lambda x:x.value_counts().index[0]))
    
    # Final feature engineering
    # Categories items ratio
    cat_features = data.columns[7:17]
    for c in cat_features:
        data[c] = data[c] / data["total_items"]
    
    # Mean delay between 2 orders
    data["order_mean_delay"] = [(y[1] - y[0]).round('1d').days if y[1] != y[0]
                                else (stop - y[0]).round('1d').days
                                for x,y in data[["first_order","last_order"]].iterrows()]
    data["order_mean_delay"] = data["order_mean_delay"] / data["nb_orders"]
    data.drop(["first_order", "last_order"], axis=1, inplace=True)
    
    # Freight ratio and total price
    data["freight_ratio"] = (round(data["total_freight"] / (data["total_spend"] + data["total_freight"]),2))
    data["total_spend"] = (data["total_spend"] + data["total_freight"])
    data.drop("total_freight", axis=1, inplace=True)
    
    # Add Haversine distance of customer state
    # Haversine distance
    olist_lat = -25.43045
    olist_lon = -49.29207
        
    geolocation['haversine_distance'] = [haversine_distance(olist_lat, olist_lon, x, y)
                                         for x, y in zip(geolocation.geolocation_lat,
                                                         geolocation.geolocation_lng)]
    data = pd.merge(data.reset_index(), geolocation[["haversine_distance"]],
                    how="left",
                    left_on="customer_state",
                    right_on="geolocation_state")
    data.drop(["customer_state"], axis=1, inplace=True)
    data.set_index("customer_unique_id", inplace=True)
    
    # complete missing values
    features_to_fill = data.isnull().sum()
    features_to_fill = list(features_to_fill[features_to_fill.values > 0].index)
    
    print(54*"_")
    print("Features complétées avec la valeur la plus fréquente :")
    print(54*"_")
    for f in features_to_fill:
        data[f] = data[f].fillna(data[f].mode()[0])
        print(f,"\t", data[f].mode()[0])
    print(54*"_")
    
    end_time = time()
    print("Durée d'execution du Feature engineering : {:.2f}s".format(end_time - start_time))
    
    return data  

def convert_to_dt(dat, cols):
    '''Function takes in a dataframe name and date columns for conversion into datetime format'''
    for col in cols:
        date_time_split = dat[col].str.split(' ', expand=True)
        dat[col] = pd.to_datetime(date_time_split[0], format='%Y-%m-%d').dt.date 
# Filling in average values for missing product values
def subst_mean(dat, cols):
    '''Function takes in name of a data frame and list of columns to substitute, na cells will be filled with mean'''
    for col in cols:
        dat[col] = dat[col].fillna(dat[col].mean())         
def classify_cat(x):
    categories = {
        'Furniture': ['office_furniture', 'furniture_decor', 'furniture_living_room', 'kitchen_dining_laundry_garden_furniture', 'bed_bath_table', 'home_comfort', 'home_comfort_2', 'home_construction', 'garden_tools', 'furniture_bedroom', 'furniture_mattress_and_upholstery'],
        'Electronics': ['auto', 'computers_accessories', 'musical_instruments', 'consoles_games', 'watches_gifts', 'air_conditioning', 'telephony', 'electronics', 'fixed_telephony', 'tablets_printing_image', 'computers', 'small_appliances_home_oven_and_coffee', 'small_appliances', 'audio', 'signaling_and_security', 'security_and_services'],
        'Fashion': ['fashio_female_clothing', 'fashion_male_clothing', 'fashion_bags_accessories', 'fashion_shoes', 'fashion_sport', 'fashion_underwear_beach', 'fashion_childrens_clothes', 'baby', 'cool_stuff'],
        'Home & Garden': ['housewares', 'home_confort', 'home_appliances', 'home_appliances_2', 'flowers', 'costruction_tools_garden', 'garden_tools', 'construction_tools_lights', 'costruction_tools_tools', 'luggage_accessories', 'la_cuisine', 'pet_shop', 'market_place'],
        'Entertainment': ['sports_leisure', 'toys', 'cds_dvds_musicals', 'music', 'dvds_blu_ray', 'cine_photo', 'party_supplies', 'christmas_supplies', 'arts_and_craftmanship', 'art'],
        'Beauty & Health': ['health_beauty', 'perfumery', 'diapers_and_hygiene'],
        'Food & Drinks': ['food_drink', 'drinks', 'food'],
        'Books & Stationery': ['books_general_interest', 'books_technical', 'books_imported', 'stationery'],
        'Industry & Construction': ['construction_tools_construction', 'construction_tools_safety', 'industry_commerce_and_business', 'agro_industry_and_commerce']
    }
    for category, keywords in categories.items():
        if x in keywords:
            return category
    return None        