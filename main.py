from __future__ import division
from collections import Counter
import math
from operator import itemgetter
import operator
import random
import numpy as np

import pandas as pd 
import sklearn
from sklearn.cluster import KMeans
from scipy.special import gammaln, psi
import pickle
import matplotlib.pyplot as plt
from preprocess import *
from detect import *
import pdb

#np.set_printoptions(threshold='nan')
#np.set_printoptions(linewidth=160)

USE_PRODUCTS = 0
USE_TIMES = 1
keyword = 'prod' if USE_PRODUCTS else 'user'

# dataname = 'itunes'
# ratings, usermap = load_itunes(USE_PRODUCTS)
breakpoint()

data = pd.read_csv( './data/amazon/amazon_network.csv')
gmm_analysis( data)
dataname = 'amazon'
ratings, usermap = load_flipkart(USE_PRODUCTS)
(rating_arr, iat_arr, ids) = process_data(ratings, dataname, USE_PRODUCTS)
(rating_arr, iat_arr) = (np.array(rating_arr), np.array(iat_arr))
# pickle.dump((rating_arr, iat_arr, ids), open('../data/%s/%s_bucketed.pickle' % (dataname, keyword), 'wb'))
# (rating_arr, iat_arr, ids) = pickle.load(open('../data/%s/%s_bucketed.pickle' % (dataname, keyword), 'rb'))

(rating_arr, iat_arr) = (rating_arr[0:5000], iat_arr[0:5000])

# Detect suspicious users given matrices containing ratings and  inter-arrival times. USE_TIMES is a boolean for whether the inter-arrival times should be used. The last parameter is the number of clusters to use. 
suspn = detect(rating_arr, iat_arr, USE_TIMES, 2)


# OUTPUT RESULTS TO FILE: it considers the top (NUM_TO_OUTPUT) most suspicious users and stores their user ids, scores, ratings and IATs in separate files.
NUM_TO_OUTPUT = 500 # number of suspicious users to output to file
susp_sorted = np.array([(x[0]) for x in sorted(enumerate(suspn), key=itemgetter(1), reverse=True)])
most_susp = susp_sorted[range(1000)]
with open('./output/%s/top%d%s_ids.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as outfile:
	with open('./output/%s/top%d%s_scores.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_scores:
		with open('./output/%s/top%d%s_ratings.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_rating:
			with open('./output/%s/top%d%s_iat.txt' % (dataname, NUM_TO_OUTPUT, keyword), 'w') as out_iat:
				for i in most_susp:
					if usermap == None:
						print ( '%s' % (ids[i],), file =outfile)
					else:
						print ('%s %s' % (ids[i], usermap[ids[i]]), file = outfile)
					print ( '%s %f' % (ids[i], suspn[i]), file =out_scores)
					print ( rating_arr[i,:], file=out_rating)
					print ( iat_arr[i,:], file =out_iat)


# PLOTTING CODE: the below code is specific to the flipkart dataset and only used to generate plots. 
bad = susp_sorted[range(100)]
bad_rate_ave_1 = np.array([0]*5, dtype=float)
bad_rate_ave_5 = np.array([0]*5, dtype=float)
bad_time_ave = np.array([0]*iat_arr.shape[1], dtype=float)

for i in range(len(suspn)):
	cur = (rating_arr[i,:] / np.sum(rating_arr[i,:]))
	if i in bad:
		if cur[0] > cur[4]: 
			bad_rate_ave_1 += cur
		else:
			bad_rate_ave_5 += cur
		bad_time_ave += (iat_arr[i,:] / np.sum(iat_arr[i,:]))

bad_rate_ave_5 = bad_rate_ave_5 / np.sum(bad_rate_ave_5)
bad_rate_ave_1 = bad_rate_ave_1 / np.sum(bad_rate_ave_1)
bad_time_ave = bad_time_ave / np.sum(bad_time_ave)

rate_sums = rating_arr.sum(axis=1)
rating_norm = rating_arr / rate_sums[:, np.newaxis]
rate_hist = rating_norm.sum(axis=0)
rate_hist = rate_hist / np.sum(rate_hist)

iat_sums = rating_arr.sum(axis=1)
time_norm = iat_arr / iat_sums[:, np.newaxis]
iat_hist = time_norm.sum(axis=0)
iat_hist = iat_hist / np.sum(iat_hist)

x = np.arange(1,6)
width = 1
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.bar(x, rate_hist, color='blue')
plt.xticks( x + width/2, [str(j) for j in x])
plt.title('Normal users', size=18)
plt.ylim(ymax=1,ymin=0)

plt.subplot(1,3,2)
plt.bar(x, bad_rate_ave_1, color='red')
plt.xticks( x + width/2, [str(j) for j in x])
plt.xlabel('Rating', size=18)
plt.title('Detected users', size=18)
plt.ylim(ymax=1,ymin=0)

plt.subplot(1,3,3)
plt.bar(x, bad_rate_ave_5, color='red')
plt.xticks(x + width/2, [str(j) for j in x])
plt.title('Detected users', size=18)
plt.ylim(ymax=1,ymin=0)
plt.savefig('./plots/flipkart_ratings_goodbad.png')

tx = range(1, len(iat_hist)+1)
fig = plt.figure(figsize=(8, 5))
plt.subplot(1,2,1)
plt.bar(tx, iat_hist, color='blue')
plt.title('Normal users', size=18)
plt.ylim(ymax=1,ymin=0)

plt.subplot(1,2,2)
plt.bar(tx, bad_time_ave, color='red')
plt.title('Detected users', size=18)
# plt.xlabel('Time between ratings (bucketized)', size=14)
plt.ylim(ymax=1,ymin=0)
fig.text(0.5, 0.02, 'IAT bucket', ha='center', size=18)

plt.savefig('./plots/flipkart_iat_goodbad.png')
