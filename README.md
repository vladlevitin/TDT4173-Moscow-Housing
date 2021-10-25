## Visualizing data
* Using online tool kepler.gl to locate datapoints based on latitude and longitude in Moscow
* Identifying flaws in dataset:
    - In training set: two buildings are not in Moscow but on St Johns and near Vladivostock (are coordinates wrong?)
    
    - In test set: one building is in Kyrgiztan, Bishkek (building id 4412)
    
    - Districs are missing in both datasets - they can possibly be retreived by searching for neighbouring buildings (130 examples in test set)
    
    - There are extreme outliers in area_total and price

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
    
* Moscow 252 metro stations: 'https://en.wikipedia.org/wiki/Module:Location_map/data/Moscow_Metro'

* Komsomolskaya metro station - linked to three railway stations

* List of Moscow metro stations: 'https://en.wikipedia.org/wiki/List_of_Moscow_Metro_stations'

* Location map+ template: 'https://en.wikipedia.org/wiki/Template:Location_map+'

Screenshot from dataset training and testing on kepler.gl

![alt text](https://github.com//vladlevitin/TDT4173-Moscow-Housing/blob/DataVisuals/visuals/kepler.gl.png?raw=true)

    
## Analysis
### Simple statistics
