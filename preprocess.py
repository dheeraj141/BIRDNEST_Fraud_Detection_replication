import numpy as np
import operator
import pickle
import math
import csv
from sklearn import mixture

TIME_LOG_BASE = 5




def gmm_analysis(X):
	lowest_bic = np.infty
	bic = []
	n_components_range = range(1, 10)
	cv_types = ['spherical','full']
	for cv_type in cv_types:
	    for n_components in n_components_range:
	        gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type, max_iter = 20)
	        gmm.fit(X)
	        bic.append(gmm.bic(X))
	        if bic[-1] < lowest_bic:
	            lowest_bic = bic[-1]
	            best_gmm = gmm

	bic = np.array(bic)
	color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
	clf = best_gmm
	bars = []
	plt.figure(figsize=(8, 6))
	spl = plt.subplot(2, 1, 1)

	for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
	    xpos = np.array(n_components_range) + .2 * (i - 2)
	    bars.append(plt.bar(xpos, bic[i * len(n_components_range):(i + 1) * len(n_components_range)], width=.2, color=color))
	
	plt.xticks(n_components_range)
	plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
	plt.title('BIC score per model')
	xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
	best_num = np.mod(bic.argmin(), len(n_components_range)) + 1
	plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
	spl.set_xlabel('Number of components')
	spl.legend([b[0] for b in bars], cv_types)

	
	splot = plt.subplot(2, 1, 2)
	Y_ = clf.predict(X)
	
	for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,color_iter)):
	    v, w = linalg.eigh(cov)
	    if not np.any(Y_ == i):
	        continue
	    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

	    angle = np.arctan2(w[0][1], w[0][0])
	    angle = 180. * angle / np.pi 
	    v = 2. * np.sqrt(2.) * np.sqrt(v)
	    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
	    ell.set_clip_box(splot.bbox)
	    ell.set_alpha(.5)
	    splot.add_artist(ell)
	
	plt.xticks(())
	plt.yticks(())
	plt.title('Selected GMM: full model,' + str(best_num) + ' components')
	plt.subplots_adjust(hspace=.35, bottom=.02)
	plt.show()
	return best_num

def load_itunes(use_products):
	csvfile = open('../data/itunes/itunes3_reviews_meta.csv', 'rb')
	outfile = open('test_out.txt', 'w')
	reader = csv.reader(csvfile, delimiter=',', escapechar='\\', quotechar=None)
	ratings = {} # {user: [(prod, time, rating)]}
	usermap = {}
	# note: users are labelled 1 through (num_users) without gaps
	for toks in reader:
		user, username, prod, rating, time = [toks[2], toks[3], toks[0], int(toks[4]), int(toks[9])]
		usermap[user] = username
		if use_products:
			user, prod = prod, user
		if user not in ratings: 
			ratings[user] = []
		ratings[user].append((prod, time, rating))
		print >> (outfile, '%s' %(user,))
	return ratings, usermap


def load_flipkart(use_products):
	fin = open('./data/amazon/amazon_network.csv', 'r')
	ratings = {} # {user: [(prod, time, rating)]}
	for line in fin:
		temp = line[:-1].split(',')
		user, prod, rating, time = [temp[0], temp[1], int(float( temp[2])), int( temp[3])]
		if use_products: user, prod = prod, user	
		if user not in ratings: 
			ratings[user] = []
		ratings[user].append((prod, time, rating))
	return ratings, None

def process_data(ratings, dataname, use_products):

	keyword = 'prod' if use_products else 'user'
	rating_arr = []
	iat_arr = []
	ids = []
	max_time_diff = -1
	for user in ratings:
		cur_ratings = sorted(ratings[user], key=operator.itemgetter(1))
		for i in range(1, len(cur_ratings)):
			time_diff = cur_ratings[i][1] - cur_ratings[i-1][1]
			max_time_diff = max(max_time_diff, time_diff)

	S = int(1 + math.floor(math.log(1 + max_time_diff, TIME_LOG_BASE)))
	for user in ratings:
		if len(ratings[user]) <= 1: continue
		rating_counts = [0] * 5
		iat_counts = [0] * S
		cur_ratings = sorted(ratings[user], key=operator.itemgetter(1))
		rating_counts[cur_ratings[0][2] - 1] += 1
		for i in range(1, len(cur_ratings)):
			time_diff = cur_ratings[i][1] - cur_ratings[i-1][1]
			iat_bucket = int(math.floor(math.log(1 + time_diff, TIME_LOG_BASE)))
			rating_counts[cur_ratings[i][2] - 1] += 1
			iat_counts[iat_bucket] += 1
		rating_arr.append(rating_counts)
		iat_arr.append(iat_counts)
		ids.append(user)

	with open('./data/%s/%s_rating_bucketed.txt' % (dataname, keyword), 'w') as rating_file:
		for row in rating_arr:
			print (' '.join([str(x) for x in row]), file = rating_file)

	with open('./data/%s/%s_iat_bucketed.txt' % (dataname, keyword), 'w') as iat_file:
		for row in iat_arr:
			print (' '.join([str(x) for x in row]), file = iat_file)

	rating_arr = np.array(rating_arr)
	iat_arr = np.array(iat_arr)
	return (rating_arr, iat_arr, ids)