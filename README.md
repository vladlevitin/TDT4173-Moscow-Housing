## Visualizing data
* Using online tool kepler.gl to locate datapoints based on latitude and longitude in Moscow
* Identifying flaws in dataset:
    

    - In training set: complete lat/long, all within 70 km distance to center.

    - There are extreme outliers in area_total and price, but this seems reasonable. The bigest flats are full top floors on buildings in central Moscow.
    
    - There are difference in distribution for area_kitchen and stories between training and test data
   
------

For assignment:

* Exploratory data analysis (at least four or more items in the list):
        * Search domain knowledge
        * Check if data is intuitive
        * Understand how data was generated
        * Explore inidividual features
        * Explore paris and groups
        * Clean up features

* Predictor used (use two or more predictors in the long notebook)

* Feature engineering (one or more feature engineering techniques - feature selection or feature extraction  in the long notebook).

* Model interpretation (use one or more model interpretation tecniques i.e PDP, feature importance, feature interaction, LIME.

-------


* Observed in data:

    - Square meter prices highest in district 0 and 7
    - Square meter prices are higher for closer distance to center, but there is still low-price apartments in these areas
    - The bigest apartments are in the center
    - z-scores are bad for apartments with the highest prices

* Price of apartment may vary according to:

    - distance to center (use distance.py to calculate distances based on latitude and longitudes)
    - distance to nearest metro station (walking time)
    - traveltime from metro station to central Moscow
    - if the metro is within one of the two ring metro lines (or outside)

* Training set values

* The following are complete:
    - price
    - area_total
    - floor
    - rooms
    - latitude
    - longitude
    - street
    - address
    - stories

* Partly complete:
    - phones
    - new
    - district (Changed to complete)
    - constructed (year)
    - elevator_without
    - elevator_person
    - elevator_service

* Missing more values:
    - kitchen_area
    - living_area
    - bathrooms_shared
    - bathrooms_private
    - material

* Missing about half or even more:
    - seller
    - ceiling (height)
    - balconies
    - loggias
    - condition
    - parking
    - garbage_chute
    - heating

* Test set missing values:

    - 4 items with wrong long/lat, get adress --> belong to district 11, lat(55.544045999) long(37.478055) = building id 4636 (two ap), 4412 (two ap)(Changed!)
    - Possibly worng long/lat? building id 4202 (one ap) - probably correct, no match on street. District set to 12 (a non existen district - not in Moscow!)
    - Possibly woring long/lat building id 8811 and 5667 (same street). Belong to district 8, lat=55.853511 long=37.38471099999 (Changed!)
    - In testing set: seven buildings are not in Moscow. Four are  St Johns, in Kirgistan (building id 4412) and two near Vladivostock. For the three ones left the coordinates are wrong (can be adjusted by searching street)
     - Districs are missing retreived by searching for neighbouring buildings street(130 examples in test set).
     - Missing 13 district values for street = 23-i-km or 25-i-km to district = 11 (there are three other apartments on this street registered with district 11)
     - Missing 2 district values for street pos kommunarka (all other buildings with this street is in district 11). Changed to district 11.
     - Missing 3 district values for street V mkr. This street has district 1 and 2, but the lat/long is different from the three missing. (Two set to district 1 - building 926, coordinates closest to district 1) Last set to district 2.
     - Two rows missing lat/long for building id 3803 (two ap) belong to street with other apartements (their lat: 55.560891000 long: 37.47376099999, district 11)(Changed!)
        
* Training set missing values:

    - Missing 68 district values for street = 23-i-km to district = 11 (there are three other apartments on this street registered with district 11). Changed to district 11.
    


* Moscow 252 metro stations: 'https://en.wikipedia.org/wiki/Module:Location_map/data/Moscow_Metro'

* Komsomolskaya metro station - linked to three railway stations

* List of Moscow metro stations: 'https://en.wikipedia.org/wiki/List_of_Moscow_Metro_stations'

* Location map+ template: 'https://en.wikipedia.org/wiki/Template:Location_map+'

Screenshot from dataset training and testing on kepler.gl

![alt text](https://github.com//vladlevitin/TDT4173-Moscow-Housing/blob/DataVisuals/visuals/kepler.gl.png?raw=true)


## Analysis
### Simple statistics
