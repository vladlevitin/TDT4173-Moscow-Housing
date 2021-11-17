import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from functions.distance import get_distance, get_distance_coordinates

"""
Cleaning dataset Moscow Housing

Test data in is pre-cleaned by finding missing Districts and missing Lat/Long

Training data in is pre-cleaned by finding missing Districts

Adjustment in this class:

Apartments outside Moscow get District = 12 (not a district in Moscow)
Test data need to be complete, even if it is "outside Moscow"

Add new variable "Distance" (from Moscow City Center)


"""

class MoscowHousing:
    def __init__(self, coordinates=None, 
                 data_train="../data/apartments_and_building_train.csv",
                 data_test="../data/apartments_and_building_test.csv", 
                 metros="../data/moscow_metros.csv", 
                 need_correction=True,
                 normalize=True,
                 features_float=["area_total", 
                                 "distance", 
                                 "area_kitchen", 
                                 "area_living", 
                                 "ceiling"]):
        self.XTest = pd.read_csv(data_test)    # original train
        self.XTrain = pd.read_csv(data_train)  # original test
        self.metros = pd.read_csv(metros)      # created from capturing web data
        self.need_correction = need_correction
        self.normalize = normalize
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.coordinates = coordinates # Coordinates to Moscows metro stations
        self.features_float = features_float
        self.features_all_s = ["building_id", "id_y", "seller", "area_total", 
                               "area_kitchen", "area_living", "floor", "rooms", 
                                "layout", "ceiling", "bathrooms_shared",
                                "bathrooms_private", "windows_court",
                                "windows_street",
                                "balconies", "loggias", "condition", "phones",
                                "new", "latitude", 
                                "longitude", "district", "street", "address",
                                "constructed",
                                "material", "stories", "elevator_without",
                                "elevator_passenger",
                                "elevator_service", "parking", "garbage_chute",
                                "heating"]
        
        self.features_all = ["building_id", "id", "seller", "area_total", 
                             "area_kitchen", "area_living", "floor", "rooms", 
                             "layout", "ceiling", "bathrooms_shared", 
                             "bathrooms_private", "windows_court", "windows_street", 
                             "balconies", "loggias", "condition", "phones", "new",
                             "latitude", 
                             "longitude", "district", "street", "address",
                             "constructed",
                             "material", "stories", "elevator_without",
                             "elevator_passenger",
                             "elevator_service", "parking", "garbage_chute",
                             "heating"]
        
        self.features_final = ["area_total", "distance", "rooms", "floor",
                          "district"]
        self.initiate()
        
    def initiate(self):
        if (self.need_correction):
        # Add removed row for building 4202 in test set (not in Moscow!)
            df1 = pd.read_csv("../data/buildings_test.csv")
            df2 = pd.read_csv("../data/apartments_test.csv")
            XTestEXTRA = pd.merge(df1, df2, how='left', 
                                  left_on=["id"], 
                                  right_on=["building_id"])
            XTestEXTRA.drop(["id_x"], axis=1)
            XTestEXTRA = XTestEXTRA[self.features_all_s]
            XTestEXTRA.fillna(-9, inplace=True)
            # Change datatypes to match target data file
            XTestEXTRA = XTestEXTRA.astype({"building_id":np.int64, "id_y":np.int64,
                                           "seller":np.int64, "floor":np.int64,
                                            "rooms":np.int64, "layout":np.int64,
                                            "bathrooms_shared":np.int64, 
                                            "bathrooms_private":np.int64,
                                            "windows_court":np.int64,
                                            "windows_street":np.int64,
                                            "balconies":np.int64, "loggias":np.int64,
                                            "condition":np.int64, "phones":np.int64,
                                            "new":np.int64, "district":np.int64,
                                            "constructed":np.int64, 
                                            "material":np.int64,
                                            "stories":np.int64,
                                            "elevator_without":np.int64,
                                            "elevator_passenger":np.int64,
                                            "elevator_service":np.int64,
                                            "parking":np.int64,
                                            "garbage_chute":np.int64,
                                            "heating":np.int64})
            XTestEXTRA = XTestEXTRA.rename(columns = {"id_y": "id"})
            Xrow_original = XTestEXTRA[XTestEXTRA["building_id"] == 4202]
            Xrow = Xrow_original.copy()
            # Set missing district for lat/long outside Moscow to 12
            Xrow.loc[Xrow["district"] == -9, "district1"] = 12
            del Xrow["district"]
            Xrow = Xrow.rename(columns = {"district1": "district"})
            # Add the missing row to the original Test dataset
            self.XTest = self.XTest.append([Xrow], ignore_index=True)
        # Check if ok
        assert self.XTest.shape[0] == 9937
        #assert self.XTest.shape[1] == 33
        
        # Replace -9 with NaN
        self.XTrain = self.XTrain.replace(-9, np.NaN)
        self.XTest = self.XTest.replace(-9, np.NaN)
        self.XTrain = self.XTrain.replace(-9., np.NaN)
        self.XTest = self.XTest.replace(-9., np.NaN)
        
        # Fix that "district" in test data is a float
        self.XTest = self.XTest.astype({"district":np.int64})
        
        # Create new features
        self.XTrain["distance"] = self.XTrain.loc[:, "latitude":"longitude"
                                                  ].apply(lambda x:                                                                                               get_distance(x.latitude, x.longitude),
                                                          axis=1)
        self.XTest["distance"] = self.XTest.loc[:, "latitude":"longitude"
                                                ].apply(lambda x:                                                                                               get_distance(x.latitude, x.longitude),
                                                        axis=1)
        # Use data captured from web scraping (used in Step_3)
        if (self.coordinates == None):
            self.coordinates = self.metros.values.tolist()
            
        self.XTrain["distance_metro"] = self.XTrain.loc[:, "latitude":"longitude"
                                                        ].apply(lambda x: get_distance_coordinates
                                                                (x.latitude, x.longitude,
                                                                self.coordinates
                                                                ), axis=1)
        self.XTest["distance_metro"] = self.XTest.loc[:, "latitude":"longitude"
                                                      ].apply(lambda x: get_distance_coordinates
                                                              (x.latitude, x.longitude,
                                                              self.coordinates
                                                              ), axis=1)


        
        
        if (self.normalize):
            # Normalize float value for "price" for y
            self.y_train = norm_features(self.y_train)
            
            # Prepare for normalizing of features
            self.X_train = self.XTrain.copy()
            self.X_test = self.XTest.copy()
            self.X_train[self.features_float] = norm_features(self.X_train
                                                              [self.features_float])
            self.X_test[self.features_float] = norm_features(self.X_test
                                                             [self.features_float])
        else:
            # Set FINAL DATA (original data IS NOT NORMALIZED)
            self.y_train = self.XTrain["price"].copy()
            self.X_train = self.XTrain.copy()
            self.X_test = self.XTest.copy()
    
    # Split training set into batches to train and test on "price"
    def get_data_split(self, features=None, 
                       test_size=0.3, random_state=42):
        if (features == None):
            features = self.features_final
        X, y = self.X_train[features].copy(), self.y_train.copy()
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size,
                                                            random_state)
        return X_train, X_test, y_train, y_test
    
    def revert_to_values(self, y_hats):
        """
        y_hats are predictions for all test examples that are 
        expressed as normalized with std (z-scores)
        
        To get the actual prices values (to return to Kaggle):
        actual values = mean prices (taken from training set) + (predictions *
        standard deviation for prices (taken form training set)
        """
        return self.XTrain["price"].mean() + (y_hats * self.XTrain["price"].std())
    
    def get_data_train(self, features=None):
        # Normalized all float variables
        if (features == None):
            return self.X_train, self.y_train
        else:
            return self.X_train[features], self.y_train
    
    def get_data_test(self, features=None):
        # Normalized all float variables
        if (features == None):
            return self.X_test
        else:
            return self.X_test[features]
        
    
    def write_results(self, file_name, predictions, revert=True):
        if revert:
            # Transform predictions as z-scores to actual values
            predictions = self.revert_to_values(predictions)
        
        result = pd.DataFrame(predictions)
        result["id"] = self.XTest["id"]
        result["price_prediction"] = result.iloc[:,0]
        # Skip first column
        result = result.iloc[:,1:]
        
        pd.DataFrame(result).to_csv(file_name, index=False)
        


# Helper functions

def norm_features(X):
    return (X - X.mean())/ (1.0 * X.std())

