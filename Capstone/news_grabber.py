import urllib2
from bs4 import BeautifulSoup

# soup = BeautifulSoup(urllib2.urlopen('http://www.timeanddate.com/worldclock/astronomy.html?n=78').read())

Ticker='GOOG'

url ='http://real-chart.finance.yahoo.com/table.csv?s='+Ticker+'&a=00&b=01&c=2015&d=08&e=30&f=2015&g=d&ignore=.csv'

soup=BeautifulSoup(urllib2.urlopen(url).read())
