import numpy as np
import math  # degrees to radians

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
    return d