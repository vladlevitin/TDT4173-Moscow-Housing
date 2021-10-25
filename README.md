## Visualizing data
* Using online tool kepler.gl to locate datapoints based on latitude and longitude in Moscow
* Identifying flaws in dataset:
    - In training set: two buildings are not in Moscow but on St Johns and near Vladivostock (are coordinates wrong?)
    
    - In test set: one building is in Kyrgiztan, Bishkek (building id 4412)
    
    - Districs are missing in both datasets - they can possibly be retreived by searching for neighbouring buildings (130 examples in test set)
    
    - There are extreme outliers in area_total and price
    
* Moscow metro stations: 'https://en.wikipedia.org/wiki/Module:Location_map/data/Moscow_Metro'

* List of Moscow metro stations: 'https://en.wikipedia.org/wiki/List_of_Moscow_Metro_stations'

* Location map+ template: 'https://en.wikipedia.org/wiki/Template:Location_map+'

Screenshot from dataset training and testing on kepler.gl

![alt text](https://github.com//vladlevitin/TDT4173-Moscow-Housing/blob/DataVisuals/visuals/kepler.gl.png?raw=true)

    
## Analysis
### Simple statistics
