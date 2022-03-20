# NTU-VFX-Project1

### Team Member
* R10922022 曾筱晴
* B09902028 曾翊綺

### Environment
* python == 3.9.6
* numpy == 1.20.1
* opencv-python == 4.5.3.56

### Run Code
```python=
python main.py [--data_path DATA_PATH] [--result_path RESULT_PATH]
               [--series_of_images SERIES_OF_IMAGES] [--shutter_time_filename SHUTTER_TIME_FILENAME]
               [--HDR_method HDR_METHOD] [--points_num POINTS_NUM] [--set_lambda SET_LAMBDA]
```
#### Optional arguments 
    * `--data_path` : Path to the directory that contains series of images.
    * `--result_path` : Path to the directory that stores all of results.
    * `--series_of_images` : The folder of a series of images that contains images and shutter time file.
    * `--shutter_time_filename` : The name of the file where shutter time information is stored.
    * `--HDR_method` : 0 : Paul Debevec's method, 1: Robertson's method.
    * `--points_num` : The number of points selected per image.
    * `--set_lambda` : The constant that determines the amount of smoothness.
    
#### Display the usage message of all arguments
```python=
python main.py --help
```