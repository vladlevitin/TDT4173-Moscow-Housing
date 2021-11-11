import numpy as np
import math                        # degrees to radians
from sklearn.utils import shuffle  # To randomize sequence of data

# The Haversine formula
# https://en.wikipedia.org/wiki/Haversine_formula
def get_distance(lat1, lon1, lat2=55.751244, lon2=37.618423):
    R = 6371 # Approximate radius of earth in km
    # Use radians
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    rlon1 = math.radians(lon1)
    rlon2 = math.radians(lon2)
    dLat = rlat2 - rlat1
    dLon = rlon2 - rlon1
    a = (np.sin(dLat/2) * np.sin(dLat/2) + np.cos(rlat1) * 
         np.cos(rlat2) * np.sin(dLon/2) * np.sin(dLon/2)
         )
    c = 2 * math.atan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d # returns km


def norm_features(X):
    return (X - X.mean())/ (1.0 * X.std())


def split_random(X, p):
    length = X.shape[0]
    X = shuffle(X)
    split_at = int(length * p)
    x_train = X[:split_at].iloc[:,1:]
    x_test = X[split_at:].iloc[:,1:]
    y_train = X[:split_at].iloc[:,:1]
    y_test = X[split_at:].iloc[:,:1]
    return x_train, y_train, x_test, y_test
    
    
def get_distance_coordinates(lat, lon, coordinates):
    shortest = np.inf
    for c in coordinates:
        #dist = haversine(c, (lat, lon), unit=Unit.METERS)
        dist = get_distance(lat, lon, c[0], c[1]) * 1000 # to get meters
        if dist < shortest:
            shortest = dist
    if shortest == np.inf:
        return 100
    return round(shortest, 2)


def PCA_plot(pca_output, c, labels=None):
    score = pca_output[:,0:2]
    coeff = np.transpose(c)
    
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.4)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, 
                     "Var"+str(i+1), color = 'g', 
                     ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, 
                     labels[i], color = 'g', 
                     ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    plt.show()