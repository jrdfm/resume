# Objective
The aim of this project is to use of maps for visualizing data. We'll be using Crime and arrest data for a Baltimore city. Checkout map.html for the visualization of the map.

## Part 1: Getting the Data


```python
import folium
import requests
import pandas

arrest_table = pandas.read_csv("https://cmsc320.github.io/files/BPD_Arrests.csv")

arrest_table = arrest_table[pandas.notnull(arrest_table["Location 1"])]
```


```python
%%capture
arrest_table["lat"], arrest_table["long"] = arrest_table["Location 1"].str.split(",").str
arrest_table["lat"] = arrest_table["lat"].str.replace("(", "").astype(float)
arrest_table["long"] = arrest_table["long"].str.replace(")", "").astype(float)
```


```python
arrest_table.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>arrest</th>
      <th>age</th>
      <th>post</th>
      <th>lat</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6.194400e+04</td>
      <td>63892.000000</td>
      <td>63801.000000</td>
      <td>63892.000000</td>
      <td>63892.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.190711e+07</td>
      <td>33.042744</td>
      <td>526.328004</td>
      <td>39.304742</td>
      <td>-76.621735</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.741072e+05</td>
      <td>11.943190</td>
      <td>263.067295</td>
      <td>0.026673</td>
      <td>0.039493</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.112686e+07</td>
      <td>0.000000</td>
      <td>111.000000</td>
      <td>39.200033</td>
      <td>-76.711492</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.126822e+07</td>
      <td>23.000000</td>
      <td>315.000000</td>
      <td>39.290457</td>
      <td>-76.650790</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.240238e+07</td>
      <td>30.000000</td>
      <td>525.000000</td>
      <td>39.303701</td>
      <td>-76.622277</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.250737e+07</td>
      <td>43.000000</td>
      <td>732.000000</td>
      <td>39.319972</td>
      <td>-76.591467</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.990540e+07</td>
      <td>87.000000</td>
      <td>945.000000</td>
      <td>39.371970</td>
      <td>-76.528438</td>
    </tr>
  </tbody>
</table>
</div>



We will drop missing values and duplicates because we have plenty of data we need to map,63892 records,and a point with missing values on a map wouldn't be so helpful. 


```python
df = arrest_table.dropna()
df = df.drop_duplicates()
```


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>arrest</th>
      <th>age</th>
      <th>post</th>
      <th>lat</th>
      <th>long</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.403500e+04</td>
      <td>54035.000000</td>
      <td>54035.000000</td>
      <td>54035.000000</td>
      <td>54035.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.189143e+07</td>
      <td>33.263311</td>
      <td>527.754918</td>
      <td>39.304667</td>
      <td>-76.622584</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.038945e+05</td>
      <td>12.079369</td>
      <td>263.614807</td>
      <td>0.026490</td>
      <td>0.039108</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.112686e+07</td>
      <td>0.000000</td>
      <td>111.000000</td>
      <td>39.202544</td>
      <td>-76.711270</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.126485e+07</td>
      <td>23.000000</td>
      <td>314.000000</td>
      <td>39.290509</td>
      <td>-76.651423</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.239698e+07</td>
      <td>30.000000</td>
      <td>526.000000</td>
      <td>39.303679</td>
      <td>-76.623862</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.250110e+07</td>
      <td>43.000000</td>
      <td>733.000000</td>
      <td>39.319058</td>
      <td>-76.592644</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.990540e+07</td>
      <td>87.000000</td>
      <td>945.000000</td>
      <td>39.371970</td>
      <td>-76.530013</td>
    </tr>
  </tbody>
</table>
</div>



We still have 54035 entries and it would be impossible to discern anything if we place all points on the map. So we will take a random sample of size 1000. We will use the random_state parameter to ensure reproducibility.  


```python
df = df.sample(n=1000,random_state=42) 
```

We will then add a column with a specfic color for each race which will make adding a color to each points easier. Each race will have color,
* Black -> black
* White -> lightgray, a white marker makes it diffcult to differentiate between sexes
* Asian -> blue
* Hispanic -> purple
* American Indian -> green
* Unknown -> orange
* NaN -> beige



```python
df['color'] = '' # set up an empty column
for j,row in df.iterrows():# iterate through our dataframe
    if row['race']== 'B': 
        df.loc[j,'color']= 'black' 
    elif row['race'] == 'W':
        df.loc[j,'color'] = 'lightgray'
    elif row['race'] == 'A':
        df.loc[j,'color'] = 'blue'
    elif row['race'] == 'I':
        df.loc[j,'color'] = 'green'
    elif row['race'] == 'H':
        df.loc[j,'color'] = 'purple'
    elif row['race'] == 'U':
        df.loc[j,'color'] = 'orange'
    elif row['race'] == 'NaN':
        df.loc[j,'color'] = 'beige'
        

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>arrest</th>
      <th>age</th>
      <th>race</th>
      <th>sex</th>
      <th>arrestDate</th>
      <th>arrestTime</th>
      <th>arrestLocation</th>
      <th>incidentOffense</th>
      <th>incidentLocation</th>
      <th>charge</th>
      <th>chargeDescription</th>
      <th>district</th>
      <th>post</th>
      <th>neighborhood</th>
      <th>Location 1</th>
      <th>lat</th>
      <th>long</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28780</th>
      <td>11272100.0</td>
      <td>41</td>
      <td>B</td>
      <td>M</td>
      <td>07/16/2011</td>
      <td>22:00:00</td>
      <td>3900 Brehms Ln</td>
      <td>87O-Narcotics (Outside)</td>
      <td>3500 Brehms La</td>
      <td>1 0573</td>
      <td>Cds: Possession-Marihuana || Poss Marijuana</td>
      <td>NORTHEASTERN</td>
      <td>415.0</td>
      <td>Belair-Edison</td>
      <td>(39.3220967987, -76.5630844959)</td>
      <td>39.322097</td>
      <td>-76.563084</td>
      <td>black</td>
    </tr>
    <tr>
      <th>37927</th>
      <td>11323019.0</td>
      <td>26</td>
      <td>B</td>
      <td>M</td>
      <td>09/16/2011</td>
      <td>10:15:00</td>
      <td>1600 N. Eutaw Pl</td>
      <td>6G-Larceny- From Bldg.</td>
      <td>300 Mcmechen St</td>
      <td>1 0521</td>
      <td>Theft Less Than $100.00 || Larceny</td>
      <td>CENTRAL</td>
      <td>133.0</td>
      <td>Madison Park</td>
      <td>(39.3063632246, -76.6293809741)</td>
      <td>39.306363</td>
      <td>-76.629381</td>
      <td>black</td>
    </tr>
    <tr>
      <th>97237</th>
      <td>12579648.0</td>
      <td>18</td>
      <td>B</td>
      <td>M</td>
      <td>10/30/2012</td>
      <td>19:15:00</td>
      <td>700 E Biddle St</td>
      <td>87-Narcotics</td>
      <td>700 E Biddle St</td>
      <td>1 0573</td>
      <td>Cds: Possession-Marihuana || Possession Marijuana</td>
      <td>EASTERN</td>
      <td>313.0</td>
      <td>Johnston Square</td>
      <td>(39.3038135255, -76.6074840977)</td>
      <td>39.303814</td>
      <td>-76.607484</td>
      <td>black</td>
    </tr>
    <tr>
      <th>80660</th>
      <td>12510330.0</td>
      <td>25</td>
      <td>B</td>
      <td>M</td>
      <td>07/09/2012</td>
      <td>19:23:00</td>
      <td>1900 Maryland Ave</td>
      <td>4E-Common Assault</td>
      <td>1900 Maryland Av</td>
      <td>1 1415</td>
      <td>Asslt-Sec Degree || Common Assault</td>
      <td>CENTRAL</td>
      <td>141.0</td>
      <td>Charles North</td>
      <td>(39.3112092048, -76.6179501743)</td>
      <td>39.311209</td>
      <td>-76.617950</td>
      <td>black</td>
    </tr>
    <tr>
      <th>17064</th>
      <td>11209415.0</td>
      <td>45</td>
      <td>B</td>
      <td>M</td>
      <td>05/03/2011</td>
      <td>17:30:00</td>
      <td>3100 W Coldspring St</td>
      <td>87-Narcotics</td>
      <td>3100 W Cold Spring La</td>
      <td>4 3550</td>
      <td>Cds:Possess-Not Marihuana || Poss Of Heroin</td>
      <td>NORTHWESTERN</td>
      <td>613.0</td>
      <td>Lucille Park</td>
      <td>(39.3370193983, -76.6729551352)</td>
      <td>39.337019</td>
      <td>-76.672955</td>
      <td>black</td>
    </tr>
  </tbody>
</table>
</div>



## Part 2: Making a Map

We will then import folliums MarkerCluster plugin which will let us cluster markers that are close to each other. The code below might be a little messy to read,especially with all the <br>s,but we need the pops to be readble and we need the breaks for that. what it's doing is just creating two sets of markers based on sex. If the incident involved a male it will have a male icon and the same for females. It will also color code the icons based on the entries color column we created based on race above. Finally, we will add the location, age, race, and charge on the popup because it will make our markers useful. On each iteration, the for loop will add a marker to the cluster.


```python
from folium.plugins import MarkerCluster

```


```python
m = folium.Map(location=[39.29, -76.61], zoom_start=11) # Set up the maps centering and zoom



cluster = folium.plugins.MarkerCluster().add_to(m) # create a cluster and add it to the map

for i,r in df.iterrows(): # iterate through the df
    if r['sex'] == 'M': # male
        folium.Marker(location=[r['lat'], r['long']],popup=folium.Popup(f'''Location: {str(r['arrestLocation'])} <br> Age: {str(r['age'])} <br> Race: {str(r['race'])}<br>  Charge: {str(r['chargeDescription'])}<br>''',max_width=len(f"name= {str(r['age'])}")*15) ,icon=folium.Icon(icon="male",color = r['color'], prefix="fa")).add_to(cluster) # add each marker to the cluster
    elif r['sex'] == 'F': # female
        folium.Marker(location=[r['lat'], r['long']], popup=folium.Popup(f'''Location: {str(r['arrestLocation'])} <br> Age: {str(r['age'])} <br> Race: {str(r['race'])}<br> Offense: {str(r['chargeDescription'])}<br>''',max_width=len(f"name= {str(r['age'])}")*15),icon=folium.Icon(icon="female",color = r['color'], prefix="fa")).add_to(cluster)# add each marker to the cluster

m.save('map.html')
m

```


