# -*- coding: utf-8 -*-
from HamSpam import NavieBays

nb=NavieBays()

nb.trainModel('/Users/jaswanthjerripothula/Desktop/SpamHam.csv')
num=nb.getAccuracy()
print(num)
predict=nb.getPrediction('I am gonna be home soon and i am Happy')
#predict=nb.getPrediction("England v Macedonia - dont miss the goals/team news")
#predict=nb.getPrediction("XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL")
print(predict)
