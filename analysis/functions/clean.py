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

class DataClean:
    def __init__(self, coordinates=None, 
                 df1 = pd.read_csv("../data/apartments_train.csv"),
                 df2 = pd.read_csv("../data/buildings_train.csv"),
                 df3 = pd.read_csv("../data/apartments_test.csv"),
                 df4 = pd.read_csv("../data/buildings_test.csv"),
                 metros = pd.read_csv("../prepared_data/moscow_metros.csv"), 
                 need_correction=True,
                 normalize=True,
                 features_float=["area_total", 
                                 "distance",
                                 "distance_metro",
                                 "area_kitchen", 
                                 "area_living", 
                                 "ceiling"]):
        self.XTrain = pd.merge(df1, df2, how='outer', left_on=["building_id"], right_on=["id"])
        self.XTest = pd.merge(df3, df4, how='outer', left_on=["building_id"], right_on=["id"])
        self.metros = metros    
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
            # Fix id names for apartment
            self.XTrain = self.XTrain.rename(columns = {"id_x": "id"})
            self.XTest = self.XTest.rename(columns = {"id_x": "id"})
            # Drop double building ids
            self.XTrain = self.XTrain.drop(["id_y"], axis=1)
            self.XTest = self.XTest.drop(["id_y"], axis=1)

            
            # Fill in missing values that are "known" both in training and testing data
            # District, Latitude, Longitude
            # Do quick correction from files preprocessed in Trifacta
            train_corr = pd.read_csv("../data/apartments_and_building_train.csv")
            test_corr  = pd.read_csv("../data/apartments_and_building_test.csv")
            
            # Sort before exchanging columns
            self.XTrain = self.XTrain.sort_values(by=["id"])
            self.XTest = self.XTest.sort_values(by=["id"])
            
            train_corr = train_corr.sort_values(by=["id"])
            test_corr = train_corr.sort_values(by=["id"])
            
            self.XTrain.loc[:,["district", "longitude", "latitude"]] = train_corr.loc[:,["district", 
                                                                                         "longitude", 
                                                                                         "latitude"]]
            self.XTest.loc[:,["district", "longitude", "latitude"]] = test_corr.loc[:,["district", 
                                                                                       "longitude", 
                                                                                       "latitude"]]
            

            
            
            """
        
            XTestEXTRA = XTestEXTRA.astype({"building_id":np.int64, "id":np.int64,
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
                                            "district":np.int64,
                                            "elevator_without":np.int64,
                                            "elevator_passenger":np.int64,
                                            "elevator_service":np.int64,
                                            "parking":np.int64,
                                            "garbage_chute":np.int64,
                                            "heating":np.int64})
           """
    
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
        self.X_test = self.XTest.copy()
        self.X_train = self.XTrain.copy()
        self.y_train = self.XTrain["price"].copy()
        
        #self.y_train = self.XTrain["price"].copy()

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
            
        # Set missing ceiling values to mean (currently ceiling is z-scores, so mean = 0)
        self.X_train["ceiling"] = self.X_train["ceiling"].fillna(0)
        self.X_test["ceiling"] = self.X_test["ceiling"].fillna(0)
        
        # Hot-encode the following features: 
        # ["seller", "layout", "condition", "new", "material", "garbage_chute", "heating"]
        hot = ["seller", "layout", "condition", "new", 
               "material", "garbage_chute", "heating", 
               "bathrooms_shared", "bathrooms_private",
               "windows_court", "windows_street",
               "balconies", "loggias",
               "phones",
               "parking"]
        
        hot_train = self.hot_encode(self.X_train[hot], hot)
        hot_test = self.hot_encode(self.X_test[hot], hot)
        
        self.X_train = pd.concat([self.X_train, hot_train], axis=1)
        self.X_test = pd.concat([self.X_test, hot_test], axis=1)
        
        # Remove original features that has been hot-encoded
        self.X_train = self.X_train.drop(hot, axis=1)
        self.X_test = self.X_test.drop(hot, axis=1)
        
        # Concatenate "elevator" into not - confirmed - nan
        elevator = ["elevator_without", "elevator_passenger", "elevator_service"]
        e_train = self.hot_encode(self.X_train[elevator], elevator)
        e_test = self.hot_encode(self.X_test[elevator], elevator)
        
        self.X_train = pd.concat([self.X_train, e_train], axis=1)
        self.X_test = pd.concat([self.X_test, e_test], axis=1)
        
        # Remove original features that has been hot-encoded
        self.X_train = self.X_train.drop(elevator, axis=1)
        self.X_test = self.X_test.drop(elevator, axis=1)
        
        # Trunkate elevator
        self.X_train["elevator"] = self.X_train.loc[:,["elevator_without_0.0", 
                                                       "elevator_passenger_1.0",
                                                       "elevator_service_1.0"
                                                       ]].apply(lambda x: x["elevator_without_0.0"] +
                                                                x["elevator_passenger_1.0"] +
                                                                x["elevator_service_1.0"] > 1, 
                                                                axis=1)
                                                    
        self.X_test["elevator"] = self.X_test.loc[:,["elevator_without_0.0", 
                                                     "elevator_passenger_1.0",
                                                     "elevator_service_1.0"
                                                     ]].apply(lambda x: x["elevator_without_0.0"] +
                                                              x["elevator_passenger_1.0"] +
                                                              x["elevator_service_1.0"] > 1, 
                                                              axis=1)
                                                    
        self.X_train["elevator_no"] = self.X_train.loc[:,["elevator_without_1.0", 
                                                          "elevator_passenger_0.0",
                                                          "elevator_service_0.0"
                                                          ]].apply(lambda x: x["elevator_without_1.0"] +
                                                                   x["elevator_passenger_0.0"] +
                                                                   x["elevator_service_0.0"] > 1,
                                                                   axis=1)
        self.X_test["elevator_no"] = self.X_test.loc[:,["elevator_without_1.0", 
                                                        "elevator_passenger_0.0",
                                                        "elevator_service_0.0"
                                                        ]].apply(lambda x: x["elevator_without_1.0"] +
                                                                 x["elevator_passenger_0.0"] +
                                                                 x["elevator_service_0.0"] > 1, 
                                                                 axis=1)
        # Remove used + nan (not kept nan)
        drop_elevator = ["elevator_without_0.0","elevator_passenger_1.0", 
                         "elevator_service_1.0","elevator_without_1.0", 
                         "elevator_passenger_0.0", "elevator_service_0.0",
                         "elevator_service_nan", "elevator_passenger_nan",
                         "elevator_without_nan"]
                                                     
        self.X_train = self.X_train.drop(drop_elevator, axis=1)
        self.X_test = self.X_test.drop(drop_elevator, axis=1)
        
        self.X_train = self.X_train.astype({"elevator":np.int8, "elevator_no":np.int8})
        self.X_test = self.X_test.astype({"elevator":np.int8, "elevator_no":np.int8})
                         
                                    
        # IMPUTE: use ["median"] from category "districts" to replace nans
        self.X_train['area_kitchen'] = self.X_train.groupby("district"
                                                            ).transform(lambda x:
                                                                        x.fillna(x.median())
                                                                        )['area_kitchen']
        self.X_test['area_kitchen'] = self.X_test.groupby("district"
                                                          ).transform(lambda x:
                                                                      x.fillna(x.median())
                                                                      )['area_kitchen']
        
        # IMPUTE: use ["median"] from category "districts" to replace nans
        self.X_train['area_living'] = self.X_train.groupby("district"
                                                           ).transform(lambda x:
                                                                       x.fillna(x.median())
                                                                       )['area_living']
        self.X_test['area_living'] = self.X_test.groupby("district"
                                                         ).transform(lambda x:
                                                                     x.fillna(x.median())
                                                                     )['area_living']
        
    def hot_encode(self, X, hot_list):
        for h in hot_list:
            df = pd.get_dummies(X[h], prefix=h, dummy_na=True)
            X = pd.concat([X, df], axis=1)
        return X

     
    
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
    
    def revert_to_values(self, z_score):
        """
        y_hats are predictions for all test examples that are 
        expressed as normalized with std (z-scores)
        
        To get the actual prices values (to return to Kaggle):
        actual values = mean prices (taken from training set) + (predictions *
        standard deviation for prices (taken form training set)
        """
        return self.XTrain["price"].mean() + (z_score * self.XTrain["price"].std())
    
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

