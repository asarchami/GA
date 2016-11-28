import feedparser

d = feedparser.parse('http://finance.yahoo.com/rss/headline?s=F')

print d['feed']['title']
print d['feed']['link']
print d.feed.subtitle
print len(d['entries'])
print d['entries'][0]['title']
print d.entries[0]['link']
