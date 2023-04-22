# pix2pix-terraform

Our project implements the GAN architecture described in the Pix2pix paper for the task of “greening” extraterrestrial imagery to earth-like satellite imagery, accomplishing this through the usage of elevation data as label maps.

While we initially considered doing image colorization, we found this task to fall in the sweet spot between originality and tractability. There exist large amounts of accessible elevation and satellite data, and the idea of using this data to terraform other planets is really fun and exciting!

### Data and Preprocessing

[Colab Earth Engine SRTM script](https://colab.research.google.com/drive/1SAF6SS1s9f5TGk_RIW9IZ6OYJN-kU2Ca?usp=sharing)

The elevation data comes from the Shuttle Radar Topography Mission, an international research effort which obtained digital elevation models for the entire world. It was accessed through Google Earth Engine at 200 meter per pixel scale (This is the same resolution as the MOLA data). Thus, we ended up with 256m x 256m = 51.2km squares.

Although we initially planned on using satellite imagery from Earth Engine as well, creating cloudless image mosaics for large areas was time-consuming and difficult. There was too much flexibility that we didn't need with regards to time scales, regions, and different satellite missions. We ultimately used images from the 2020 [Sentinel-2 Cloudless](https://s2maps.eu/) Mosaic, licenced under CC BY-NC-SA 4.0. It has an amazing API that lets you download cloudless imagery for any geographic coordinates in a GET request, provided that they are 4096x4096 pixels or less. The scale was also set to be 200 meters per pixels and 256 x 256 outputs, which we retrieved regionally from:

- The Bolivian Andes 
- Southern California 
- Italy
- Nepal
- Pacific Northwest
- Rhode Island, New York
- Sichuan Province, China
- [TODO] add other places and coordinates
  
Both data sources use EPSG:4326, which considered the standard geographic coordinate system. However, simply specifying a region of equal latitude and longitude will not result in a square, because longitudinal lines converge as the latitude increases. To ensure that the areas were truly 51.2km by 51.2km, we adjusted the longitude by the Haversine formula.

QGIS was used to uniformly colorize the elevation data from the 0 to 9000m range, and both were resized and converted to JPG format for use in the model.

