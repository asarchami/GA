
from bs4 import BeautifulSoup
import urllib2
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from time import sleep
import os.path
import re
import glob
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV, LassoCV
# from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error, silhouette_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV

from sklearn.externals import joblib

import pydot
from IPython.display import Image

import seaborn as sns
import matplotlib.pyplot as plt


def get_companies():
    # This function downloads list of S&P 500 from wikipedia and
    # stores them in a csv file. If there is such a file it just loads it
    # and return companies information as Panda DataFrame
    companies = None
    if not os.path.exists('datasets/s_p_500.csv'):
        companies = pd.read_html(
            'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
            attrs={"class": 'wikitable sortable'}, header=0)[0]
        companies.to_csv('datasets/s_p_500.csv', encoding="utf-8")
    else:
        companies = pd.read_csv('datasets/s_p_500.csv', index_col=0)

    companies.columns = [x.strip().replace(
        ' ', '_') for x in companies.columns]
    companies.Ticker_symbol = companies.Ticker_symbol.apply(
        lambda x: x.replace('-', ''))
    return companies


def get_article(link, title, date):
    # This function gets article from the provided link
    article = {}
    article['Url'] = "http://www.reuters.com/"+link
    article['Title'] = title
    article['Symbol'] = symbol
    article['Date'] = date
    soup = BeautifulSoup(urllib2.urlopen(article['Url']), "lxml")
    article['Article'] = soup.find_all(
        'span', {"id": "article-text"})[0].text.replace('\n', ' ')
    article['Time'] = soup.find_all(
        'span', {"class": "timestamp"})[0].text
    # this is needed for not being blocked!!!
    sleep(random.randint(1, 6)/10.0)
    return link


def symbol_news(url):
    # This function gets links of articles in each page (url)
    links = []
    news = BeautifulSoup(
        urllib2.urlopen(url), "lxml").find_all('div', {"id": "companyNews"})
    for i in xrange(2):
        for feature in news[i].find_all('h2'):
            a = feature.find('a')
            if a.has_attr('href'):
                d = re.search('[0-9]{8}', url).group(0)
                day = datetime.strptime(d, '%m%d%Y').date()
                links.append(get_article(a['href'], a.text, day))
    sleep(random.randint(1, 6)/10.0)
    return links


def get_articles_for_symbol(symbol,
                            start_date=date(2014, 1, 1),
                            end_date=date(2014, 2, 1)):
    # gets articles for a symbol from start_date to end_date
    days = [start_date + timedelta(n) for n in range(
        (end_date - start_date).days)]
    articles = []
    for day in days:
        url = "http://www.reuters.com/finance/stocks/companyNews?symbol={}&date={}".format(
            symbol, day.strftime('%m%d%Y'))
        failed_urls = []
        try:
            articles += symbol_news(url)
        except (urllib2.URLError, IOError, httplib.HTTPException) as e:
            print e.args
            print url
            failed_urls.append(url)
            pass
        if len(failed_urls) > 0:
            failed_urls = pd.DataFrame(failed_urls)
            failed_urls['Symbol'] = symbol
            if not os.path.exists('datasets/companies/failed_urls.csv'):
                failed_urls.to_csv(
                    'datasets/companies/failed_urls.csv', encoding='utf8')
            else:
                all_failed = pd.read_csv(
                    'datasets/companies/failed_urls.csv', index_col=0)
                all_failed = pd.concat([all_failed, failed_urls], axis=1)
                all_failed.to_csv(
                    'datasets/companies/failed_urls.csv', encoding='utf8')
    articles = pd.DataFrame(articles)
    return articles


def download_articles(companies):
    # This function downloads articles from Reuters
    # and save them in datasets/companies
    for symbol in companies['Ticker_symbol']:
        print symbol
        if not os.path.exists('datasets/companies/{}.csv'.format(symbol)):
            article = get_articles_for_symbol(symbol, end_date=date.today())
            article.to_csv(
                'datasets/companies/{}.csv'.format(symbol), encoding='utf8')
    return True


def get_articles(companies):
    # this function goes through files stored in datasets/companies
    # and gets news about all S&P 500 companies.
    articles = []
    not_found = []
    # if not os.path.exists('datasets/articles.csv'):
    for symbol in companies['Ticker_symbol']:
        if not os.path.exists('datasets/companies/{}.csv'.format(symbol)):
            not_found.append(symbol)
        else:
            article = pd.read_csv(
                'datasets/companies/{}.csv'.format(symbol),
                encoding='utf8', index_col=0)
            articles.append(article)
    articles = pd.concat(articles, axis=0)

    articles['Date'] = pd.to_datetime(articles['Date'], format='%Y-%m-%d')
    print not_found
    return articles


def update_sentimentalized_articles(articles):
    filenames = sorted(
        glob.glob('datasets/sentimentalized/articles_part*.csv'))
    s_articles = [pd.read_csv(
        filename, encoding='utf8', index_col=0) for filename in filenames]
    s_articles = pd.concat(s_articles, axis=0)
    print articles.shape, s_articles.shape
    not_s_syms = [x for x in list(
        articles.Symbol.unique()) if x not in list(s_articles.Symbol.unique())]
    print "need to sentimentalize {} symbols".format(len(not_s_syms))

    if len(not_s_syms) > 0:
        not_s_articles = articles[articles['Symbol'].isin(not_s_syms)]
        sid = SentimentIntensityAnalyzer()
        lambda_sentimenter = lambda x: pd.Series(
            [sid.polarity_scores(x['Article'])['compound'],
             sid.polarity_scores(x['Article'])['neg'],
             sid.polarity_scores(x['Article'])['neu'],
             sid.polarity_scores(x['Article'])['pos']])
        sents = not_s_articles.apply(lambda_sentimenter, axis=1)
        sents.columns = ['compound', 'neg', 'neu', 'pos']
        sents = sents.reset_index().drop('index', axis=1)
        not_s_articles = not_s_articles.reset_index().drop('index', axis=1)
        not_s_articles = pd.concat([not_s_articles, sents], axis=1)
        s_articles = pd.concat([s_articles, not_s_articles], axis=0)
        not_s_articles.to_csv(
            'datasets/sentimentalized/articles_part_{}.csv'.format(
                len(filenames)+1),
            encoding='utf8')
        print 'updated sentiments'
        return s_articles

    else:
        print 'Nothing updated'
        return s_articles


def get_sentimentalized_articles(companies):
    articles = get_articles(companies)
    return update_sentimentalized_articles(articles)


def get_quote(symbol, start_date='1/1/2014',
              end_date=date.today().strftime('%m/%d/%Y')):
    symbol = symbol.replace('.N', '')
    try:
        sleep(2)
        ret = data.DataReader(symbol,
                              data_source='yahoo',
                              start=start_date, end=end_date)
    except (urllib2.URLError, IOError) as e:
        print symbol, e.args
        sleep(10)
        ret = data.DataReader(symbol,
                              data_source='google',
                              start=start_date, end=end_date)
        ret['Adj Close'] = None
        pass
    ret['Symbol'] = symbol
    return ret


def update_quotes(articles, reset=False):
    not_downloaded = False
    quotes = False
    if not os.path.exists('datasets/daily_quotes.csv'):
        not_downloaded = list(articles.Symbol.unique())
        print 'Its empty!'
    else:
        quotes = pd.read_csv(
            'datasets/daily_quotes.csv',
            encoding='utf8', index_col=0)
        not_downloaded = [x for x in list(
            articles.Symbol.unique()) if x not in list(quotes.Symbol.unique())]
        print "Available symbol quotes: ", len(list(quotes.Symbol.unique()))

    for symbol in not_downloaded:
        quote = get_quote(symbol)
        if not quotes.empty:
            quotes = pd.concat([quotes, quote], axis=0)
        else:
            quotes = quote
        quotes.to_csv('datasets/daily_quotes.csv', encoding='utf8')

    print "New downloaded symbol quotes: ", len(not_downloaded)
    quotes = pd.read_csv('datasets/daily_quotes.csv', encoding='utf8')
    return quotes


def merge_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z


def CV(target):
    return StratifiedKFold(target, n_folds=3,
                           shuffle=True, random_state=41)


def grid_search(model, params, cv):
    return GridSearchCV(estimator=model,
                        param_grid=params,
                        cv=cv)


def regression_results(model, x_true, y_true, y_pred):
    ret = {}
    ret["Explained variance regression score"] = explained_variance_score(
        y_true, y_pred, multioutput='uniform_average')
    ret["Mean Absolute Error"] = mean_absolute_error(
        y_true, y_pred)
    ret["Model socre"] = model.score(
        x_true, y_true)
    return ret


def draw_feature_importance(model, data):
    importances = model.feature_importances_
    std = np.std(
        [tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    feature_names = data.columns
    # Plot the feature importances of the model
    plt.figure(figsize=(20, 15))
    plt.title("Feature importances")
    plt.bar(range(data.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(data.shape[1]), feature_names[indices], rotation=90)
    plt.xlim([-1, data.shape[1]])
    return plt


def model_test_metrics(model, data, target):
    ret = {}
    ret["Test Data score"] = model.fit(data, target).score(data, target)
    return merge_dicts(ret, regression_results(target, model.predict(data)))


def evaluate_model(model, data, target,
                   params=None, draw_features=True, verbose=True):
    train_data, test_data, train_target, test_target = train_test_split(
        data, target)
#     cv=CV(train_target)
    cv = 3
    ret = {}
    if params:
        grid = grid_search(model, params, cv)

        grid.fit(train_data, train_target)
        model = grid.best_estimator_
    else:
        model.fit(train_data, train_target)

    s = cross_val_score(model, train_data, train_target, cv=cv, n_jobs=-1)
    ret["Mean cross validation score"] = s.mean()
    predictions = model.predict(test_data)
    ret = merge_dicts(ret, regression_results(
        model, test_data, test_target, predictions))

    if draw_features:
        draw_feature_importance(model, train_data)
    return model, ret


def linear_regression(data, target):
    params = {
        'fit_intercept': [False, True]
    }
    return evaluate_model(LinearRegression(n_jobs=-1),
                          data, target,
                          draw_features=False,
                          params=params)


def ridge_cv(data, target):
    return evaluate_model(
        RidgeCV(alphas=[0.1, .5, 1.0, 5.0, 10.0, 100.0], normalize=True),
        data, target, draw_features=False)


def random_forest(data, target, draw_features=False):
    params = {
        'n_estimators': [10, 20, 50, 100, 200],
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 10, 25, 50, 100],
    }
    return evaluate_model(RandomForestRegressor(n_jobs=-1),
                          data, target,
                          params=params, draw_features=draw_features)


def extra_trees(data, target, draw_features=False):
    params = {
        'n_estimators': [10, 20, 50, 100, 200],
        'max_depth': [1, 2, 3, 4, 5],
        'min_samples_split': [2, 10, 25, 50, 100],
    }
    return evaluate_model(ExtraTreesRegressor(n_jobs=-1),
                          data, target,
                          params=params, draw_features=draw_features)


def print_model_report(report, score=None, model=None):
    if model:
        print model, "\n"
    for title, value in report.iteritems():
        print "{:<50}{}".format(title, value)
    if score:
        print "\n", "{:<50}{}".format('score for test data', score)


def plot_predicted_measured(ax, y_real, predictions, fontsize=24, title=''):
    ax.scatter(y_real, predictions)
    ax.plot([predictions.min(), predictions.max()],
            [predictions.min(), predictions.max()], 'k--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(title, fontsize=fontsize)


def set_style():
    plt.style.use(['seaborn-white', 'seaborn-paper'])


def plot_comparisions(y_real, y_lr, y_rd, y_rf, y_et):
    set_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    plot_predicted_measured(
        ax1, y_real, y_lr, fontsize=16, title='Linear Regression')
    plot_predicted_measured(
        ax2, y_real, y_rd, fontsize=16, title='RidgeCV')
    plot_predicted_measured(
        ax3, y_real, y_rf, fontsize=16, title='Random Forest')
    plot_predicted_measured(
        ax4, y_real, y_et, fontsize=16, title='Extra Trees')

    plt.show()
