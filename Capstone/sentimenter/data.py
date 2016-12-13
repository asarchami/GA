
from bs4 import BeautifulSoup
import urllib2
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from time import sleep
import os.path
import re


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

    companies.columns = [x.strip().replace(' ', '_') for x in companies.columns]
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
    days = [start_date + timedelta(n) for n in range((end_date - start_date).days)]
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