import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.linear_model import BayesianRidge
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
    def __init__(self, 
                 coordinates=None,
                 features_float=None,
                 df1 = pd.read_csv("../data/apartments_train.csv"),
                 df2 = pd.read_csv("../data/buildings_train.csv"),
                 df3 = pd.read_csv("../data/apartments_test.csv"),
                 df4 = pd.read_csv("../data/buildings_test.csv"),
                 metros = pd.read_csv("../data/moscow_metros.csv"),
                 universities = pd.read_csv("../data/universities.csv"),
                 golf = pd.read_csv("../data/golf_courses.csv"),
                 parks = pd.read_csv("../data/parks.csv"),
                 airports = pd.read_csv("../data/airports.csv"),
                 shopping = pd.read_csv("../data/shopping_centers.csv"),
                 prisons = pd.read_csv("../data/prisons.csv"),
                 need_correction=True,
                 normalize=True):
        self.XTrain = pd.merge(df1, df2, how='outer', left_on=["building_id"], right_on=["id"])
        self.XTest = pd.merge(df3, df4, how='outer', left_on=["building_id"], right_on=["id"])
        self.metros = metros
        self.park = parks
        self.golf = golf
        self.airport = airports
        self.university = universities
        self.shopping = shopping
        self.prison = prisons
        self.need_correction = need_correction
        # Setting IMPUTER
        self.imputer = IterativeImputer(estimator=BayesianRidge(),
                                        random_state=42,
                                        imputation_order='ascending', 
                                        max_iter=100,
                                        tol=1e-5)
        self.normalize = normalize
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.coordinates = coordinates # Placeholder for coordinates to Moscows metro stations
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
        # Merge the changes done in Trifacta by adding missing long/lat and districts
        if (self.need_correction):
            # Set "id" for apartment in the merged dataset
            self.XTrain = self.XTrain.rename(columns = {"id_x": "id"})
            self.XTest = self.XTest.rename(columns = {"id_x": "id"})
            # Drop double building ids
            self.XTrain = self.XTrain.drop(["id_y"], axis=1)
            self.XTest = self.XTest.drop(["id_y"], axis=1)

            # Fill in missing values that are "known" both in training and testing data
            # District, Latitude, Longitude
            # Do quick correction from files preprocessed in Trifacta
            train_corr = pd.read_csv("../data/apartments_and_building_train.csv")
            # One item was removed, id=4202
            test_corr  = pd.read_csv("../data/apartments_and_building_test.csv")
           
            
            # Fix missing row in test_corr
            Xrow_original = self.XTest[self.XTest["building_id"] == 4202]
            Xrow = Xrow_original.copy()
            # Set missing district for lat/long outside Moscow to 12
            Xrow.loc[Xrow["district"] == -9, "district1"] = 12
            del Xrow["district"]
            Xrow = Xrow.rename(columns = {"district1": "district"})
            # Add the missing row to the original Test dataset
            test_corr = test_corr.append([Xrow], ignore_index=True)
            
            # Sort before exchanging columns
            self.XTrain = self.XTrain.sort_values(by=["id"])
            self.XTest = self.XTest.sort_values(by=["id"])
            
            train_corr = train_corr.sort_values(by=["id"])
            test_corr = train_corr.sort_values(by=["id"])  
            
            # Make sure the number of examples are the same
            assert self.XTrain.shape[0] == train_corr.shape[0]
            assert self.XTest.shape[0] == 9937
            
            # Replace all the corrected columns
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
        
        # Use data for metro coordinates are captured from web scraping (used in Step_3)
        if (self.coordinates == None):
            self.coordinates = self.metros.values.tolist()
            
        # Set distance to closest metro station
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
        
        # Set distance to closest park
        self.coordinates = self.park.values.tolist()
        
        self.XTrain["park"] = self.XTrain.loc[:, "latitude":"longitude"
                                              ].apply(lambda x: get_distance_coordinates
                                              (x.latitude, x.longitude,
                                               self.coordinates
                                               ), axis=1)
        self.XTest["park"] = self.XTest.loc[:, "latitude":"longitude"
                                            ].apply(lambda x: get_distance_coordinates
                                            (x.latitude, x.longitude,
                                             self.coordinates
                                             ), axis=1)
        
        
        # Set distance to closest golf court
        self.coordinates = self.golf.values.tolist()
        
        self.XTrain["golf"] = self.XTrain.loc[:, "latitude":"longitude"
                                              ].apply(lambda x: get_distance_coordinates
                                              (x.latitude, x.longitude,
                                               self.coordinates
                                               ), axis=1)
        
        self.XTest["golf"] = self.XTest.loc[:, "latitude":"longitude"
                                            ].apply(lambda x: get_distance_coordinates
                                            (x.latitude, x.longitude,
                                             self.coordinates
                                             ), axis=1)
        
        # Set distance to closest airport
        self.coordinates = self.airport.values.tolist()
        
        self.XTrain["airport"] = self.XTrain.loc[:, "latitude":"longitude"
                                                 ].apply(lambda x: get_distance_coordinates
                                                 (x.latitude, x.longitude,
                                                  self.coordinates
                                                  ), axis=1)
        
        self.XTest["airport"] = self.XTest.loc[:, "latitude":"longitude"
                                               ].apply(lambda x: get_distance_coordinates
                                                (x.latitude, x.longitude,
                                                 self.coordinates
                                                 ), axis=1)
        
        
        # Set distance to closest university court
        self.coordinates = self.university.values.tolist()
        
        self.XTrain["university"] = self.XTrain.loc[:, "latitude":"longitude"
                                                    ].apply(lambda x: get_distance_coordinates
                                                    (x.latitude, x.longitude,
                                                     self.coordinates
                                                     ), axis=1)
        
        self.XTest["university"] = self.XTest.loc[:, "latitude":"longitude"
                                                  ].apply(lambda x: get_distance_coordinates
                                                  (x.latitude, x.longitude,
                                                   self.coordinates
                                                   ), axis=1)
        
        
        # Set distance to closest shoping center
        self.coordinates = self.shopping.values.tolist()
        
        self.XTrain["shopping"] = self.XTrain.loc[:, "latitude":"longitude"
                                                  ].apply(lambda x: get_distance_coordinates
                                                  (x.latitude, x.longitude,
                                                   self.coordinates
                                                   ), axis=1)
        self.XTest["shopping"] = self.XTest.loc[:, "latitude":"longitude"
                                                ].apply(lambda x: get_distance_coordinates
                                                (x.latitude, x.longitude,
                                                 self.coordinates
                                                 ), axis=1)
        
        # Set distance to closest prison
        self.coordinates = self.prison.values.tolist()
        
        self.XTrain["prison"] = self.XTrain.loc[:, "latitude":"longitude"
                                                ].apply(lambda x: get_distance_coordinates
                                                (x.latitude, x.longitude,
                                                 self.coordinates
                                                 ), axis=1)
        self.XTest["prison"] = self.XTest.loc[:, "latitude":"longitude"
                                              ].apply(lambda x: get_distance_coordinates
                                              (x.latitude, x.longitude,
                                               self.coordinates
                                               ), axis=1)
        
        
        # Copy original data before further feature modifications
        self.X_test = self.XTest.copy()
        self.X_train = self.XTrain.copy()
        self.y_train = self.XTrain["price"].copy()
        
        """   
        # Set missing ceiling values to mean (currently ceiling is z-scores, so mean = 0)
        self.X_train["ceiling"] = self.X_train["ceiling"].fillna(0)
        self.X_test["ceiling"] = self.X_test["ceiling"].fillna(0)
        """

        # Hot-encode the following features: 
        # ["seller", "layout", "condition", "new", "material", "garbage_chute", "heating"]
        hot = ["seller", "layout", "condition", "new", 
               "material", "garbage_chute", "heating", 
               "bathrooms_shared", "bathrooms_private",
               "windows_court", "windows_street",
               "balconies", "loggias",
               "phones",
               "parking",
               "district"]
        
        hot_train = self.hot_encode(self.X_train[hot], hot, na=True)
        hot_test = self.hot_encode(self.X_test[hot], hot, na=True)
        
        self.X_train = pd.concat([self.X_train, hot_train], axis=1)
        self.X_test = pd.concat([self.X_test, hot_test], axis=1)
        
        # Remove the original features that now have been hot-encoded
        self.X_train = self.X_train.drop(hot, axis=1)
        self.X_test = self.X_test.drop(hot, axis=1)
        
        # Concatenate "elevator" into not - confirmed - nan
        elevator = ["elevator_without", "elevator_passenger", "elevator_service"]
        
        e_train = self.hot_encode(self.X_train[elevator], elevator, na=False)
        e_test = self.hot_encode(self.X_test[elevator], elevator, na=False)
        
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
                                                                x["elevator_service_1.0"] > 0.9, 
                                                                axis=1)
                                                    
        self.X_test["elevator"] = self.X_test.loc[:,["elevator_without_0.0", 
                                                     "elevator_passenger_1.0",
                                                     "elevator_service_1.0"
                                                     ]].apply(lambda x: x["elevator_without_0.0"] +
                                                              x["elevator_passenger_1.0"] +
                                                              x["elevator_service_1.0"] > 0.9, 
                                                              axis=1)
                                                    
        self.X_train["elevator_no"] = self.X_train.loc[:,["elevator_without_1.0", 
                                                          "elevator_passenger_0.0",
                                                          "elevator_service_0.0"
                                                          ]].apply(lambda x: x["elevator_without_1.0"] +
                                                                   x["elevator_passenger_0.0"] +
                                                                   x["elevator_service_0.0"] > 0.9,
                                                                   axis=1)
        
        self.X_test["elevator_no"] = self.X_test.loc[:,["elevator_without_1.0", 
                                                        "elevator_passenger_0.0",
                                                        "elevator_service_0.0"
                                                        ]].apply(lambda x: 
                                                                 x["elevator_without_1.0"] +
                                                                 x["elevator_passenger_0.0"] +
                                                                 x["elevator_service_0.0"] > 0.9,
                                                                 axis=1)

        # Trunkate balconies, loggias, bathrooms_shared, bathrooms_private
        self.X_train["balconies_yes"] = self.X_train.loc[:,["balconies_1.0", "balconies_2.0",
                                                         "balconies_3.0", "balconies_4.0"
                                                         ]].apply(lambda x: 
                                                                  x["balconies_1.0"] +
                                                                  x["balconies_2.0"] +
                                                                  x["balconies_3.0"] + 
                                                                  x["balconies_4.0"] > 0.9, 
                                                                  axis=1)
                                                                 
                                                                 
        self.X_train["loggias_yes"] = self.X_train.loc[:,["loggias_1.0", "loggias_2.0",
                                                       "loggias_3.0", "loggias_4.0"
                                                       ]].apply(lambda x: 
                                                                x["loggias_1.0"] +
                                                                x["loggias_2.0"] +
                                                                x["loggias_3.0"] + 
                                                                x["loggias_4.0"] 
                                                                > 0.9, axis=1)


        self.X_train["bathrooms_shared_yes"] = self.X_train.loc[:,["bathrooms_shared_1.0",                                                                               "bathrooms_shared_2.0",
                                                                "bathrooms_shared_3.0",
                                                                "bathrooms_shared_4.0"
                                                                ]].apply(lambda x:
                                                                         x["bathrooms_shared_1.0"] +
                                                                         x["bathrooms_shared_2.0"] +
                                                                         x["bathrooms_shared_3.0"] + 
                                                                         x["bathrooms_shared_4.0"]
                                                                         > 0.9, axis=1)
                                                                 
        self.X_train["bathrooms_private_yes"] = self.X_train.loc[:,["bathrooms_private_1.0",                                                                              "bathrooms_private_2.0",
                                                                 "bathrooms_private_3.0",
                                                                 "bathrooms_private_4.0"
                                                                 ]].apply(lambda x:
                                                                          x["bathrooms_private_1.0"] +
                                                                          x["bathrooms_private_2.0"] +
                                                                          x["bathrooms_private_3.0"] + 
                                                                          x["bathrooms_private_4.0"] 
                                                                          > 0.9, axis=1)
        
        self.X_test["balconies_yes"] = self.X_test.loc[:,["balconies_1.0", "balconies_2.0",
                                                       "balconies_3.0", "balconies_4.0"
                                                       ]].apply(lambda x: 
                                                                x["balconies_1.0"] +
                                                                x["balconies_2.0"] +
                                                                x["balconies_3.0"] + 
                                                                x["balconies_4.0"] > 0.9, 
                                                                axis=1)
                                                                 
                                                                 
        self.X_test["loggias_yes"] = self.X_test.loc[:,["loggias_1.0", "loggias_2.0",
                                                     "loggias_3.0", "loggias_4.0"
                                                     ]].apply(lambda x: 
                                                              x["loggias_1.0"] +
                                                              x["loggias_2.0"] +
                                                              x["loggias_3.0"] + 
                                                              x["loggias_4.0"] > 0.9, 
                                                              axis=1)


        self.X_test["bathrooms_shared_yes"] = self.X_test.loc[:,["bathrooms_shared_1.0",                                                                               "bathrooms_shared_2.0",
                                                              "bathrooms_shared_3.0",
                                                              "bathrooms_shared_4.0"
                                                              ]].apply(lambda x:
                                                                       x["bathrooms_shared_1.0"] +
                                                                       x["bathrooms_shared_2.0"] +
                                                                       x["bathrooms_shared_3.0"] + 
                                                                       x["bathrooms_shared_4.0"] > 0.9, 
                                                                       axis=1)
                                                                 
        self.X_test["bathrooms_private_yes"] = self.X_test.loc[:,["bathrooms_private_1.0",                                                                              "bathrooms_private_2.0",
                                                               "bathrooms_private_3.0",
                                                               "bathrooms_private_4.0"
                                                               ]].apply(lambda x:
                                                                        x["bathrooms_private_1.0"] +
                                                                        x["bathrooms_private_2.0"] +
                                                                        x["bathrooms_private_3.0"] + 
                                                                        x["bathrooms_private_4.0"] 
                                                                        > 0.9, axis=1)
        # Remove used (except the hot-encoded nan features)
        drop_h = ["elevator_without_0.0","elevator_passenger_1.0", 
                  "elevator_service_1.0","elevator_without_1.0", 
                  "elevator_passenger_0.0", "elevator_service_0.0",
                  "bathrooms_private_1.0", "bathrooms_private_2.0",
                  "bathrooms_private_3.0", "bathrooms_private_4.0",
                  "bathrooms_shared_1.0", "bathrooms_shared_2.0",
                  "bathrooms_shared_3.0", "bathrooms_shared_4.0",
                  "loggias_1.0", "loggias_2.0","loggias_3.0", 
                  "loggias_4.0", "balconies_1.0", "balconies_2.0",
                  "balconies_3.0", "balconies_4.0"]
                                                     
        self.X_train = self.X_train.drop(drop_h, axis=1)
        self.X_test = self.X_test.drop(drop_h, axis=1)
        
        self.X_train = self.X_train.astype({"elevator":np.int8, 
                                            "elevator_no":np.int8,
                                            "balconies_yes":np.int8,
                                            "loggias_yes":np.int8,
                                            "bathrooms_shared_yes":np.int8,
                                            "bathrooms_private_yes":np.int8})
        
        self.X_test = self.X_test.astype({"elevator":np.int8, 
                                          "elevator_no":np.int8,
                                          "balconies_yes":np.int8,
                                          "loggias_yes":np.int8,
                                          "bathrooms_shared_yes":np.int8,
                                          "bathrooms_private_yes":np.int8})
        
        
                         
        """    
        # IMPUTE: area_kitchen
        self.X_train["area_kitchen"] = self.X_train.groupby("building_id"
                                                            ).transform(lambda x:
                                                                        x.fillna(x.median())
                                                                        )["area_kitchen"]
        self.X_test["area_kitchen"] = self.X_test.groupby("building_id"
                                                          ).transform(lambda x:
                                                                      x.fillna(x.median())
                                                                      )["area_kitchen"]
                                    
        # IMPUTE: use ["median"] from category "districts" to replace nans
        self.X_train["area_kitchen"] = self.X_train.groupby("district"
                                                            ).transform(lambda x:
                                                                        x.fillna(x.median())
                                                                        )["area_kitchen"]
        
        self.X_test["area_kitchen"] = self.X_test.groupby("district"
                                                          ).transform(lambda x:
                                                                      x.fillna(x.median())
                                                                      )["area_kitchen"]
        
        # IMPUTE use median from category building_id to first replace nans
        self.X_train["area_living"] = self.X_train.groupby("building_id"
                                                           ).transform(lambda x:
                                                                       x.fillna(x.median())
                                                                       )["area_living"]
        
        self.X_test["area_living"] = self.X_test.groupby("building_id"
                                                         ).transform(lambda x:
                                                                     x.fillna(x.median())
                                                                     )["area_living"]
        
        # IMPUTE: use ["median"] from category "districts" to replace nans
        self.X_train["area_living"] = self.X_train.groupby("district"
                                                           ).transform(lambda x:
                                                                       x.fillna(x.median())
                                                                       )['area_living']
        
        self.X_test["area_living"] = self.X_test.groupby("district"
                                                         ).transform(lambda x:
                                                                     x.fillna(x.median())
                                                                    )['area_living']
        """ 

        
        if (self.normalize):
            # Normalize float value for "price" for y (create z-scores) NOT HERE!
            #self.y_train = norm_features(self.y_train)
            # Prepare for normalizing of features
            # Pass the float features as a list [] in constructor
            self.X_train[self.features_float] = norm_features(self.X_train
                                                              [self.features_float])
            self.X_test[self.features_float] = norm_features(self.X_test
                                                             [self.features_float])
            
        # Iterative impute of feature "constructed"
        features_impute = ["constructed", "area_kitchen", "area_living", "ceiling"]
        imputer = IterativeImputer(random_state=42)
        
        imputed1 = imputer.fit_transform(self.X_train[features_impute])
        self.X_train[features_impute] = pd.DataFrame(imputed1, columns=features_impute)
        
        imputed2 = imputer.fit_transform(self.X_test[features_impute])
        self.X_test[features_impute] = pd.DataFrame(imputed2, columns=features_impute)
        
        
        
        
        
    def hot_encode(self, X, hot_list, na=True):
        for h in hot_list:
            df = pd.get_dummies(X[h], prefix=h, dummy_na=na)
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

def revert(X, z_score):
        """
        X ........... is the training set containing price data
        
        z_score ..... are predictions for all test examples that are 
                      expressed as normalized with standard deviation
        """
        return X["price"].mean() + (z_score * X["price"].std())
    
# If price is first np.log(price) and then normalized before analysis
def revert_ln(X, pred):
    ln_X = np.log(X)
    ln_X_mean = ln_X.mean()
    ln_X_std = ln_X.std()
    # pred are normalized/standardized np.log(y) predictions
    return np.exp(ln_X_mean + (ln_X_std * pred))
    
def revert_log(log_score):
    return np.exp(log_score)

def write_predictions(file_name, X_train, X_test, predictions, log=False, reverting=True):
    if (log):
        if reverting:
            predictions = revert_log(predictions)
    else:
        if reverting:
                # Transform predictions as z-scores to actual values
                predictions = revert(X_train, predictions)
                
    result = pd.DataFrame(predictions)
    result["id"] = X_test["id"]
    result["price_prediction"] = result.iloc[:,0]
    # Skip first column
    result = result.iloc[:,1:]
    # Write the file
    pd.DataFrame(result).to_csv(file_name, index=False)