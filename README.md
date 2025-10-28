# AmBe Source Watcher - Videocamera Surveillance

This code is used to implement a person identificator using a deep neural network model. It receives the feed from a camera using the RTSP stream. 


# Dependencies

You should make sure you download the following files to this directory

[MobileNetSSD_deploy.prototxt](https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view?resourcekey=0-1Lpfs4EvGDeCQz12AF64hQ)

[MobileNetSSD_deploy.caffemodel](https://gist.github.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f#file-mobilenetssd_deploy-prototxt)

Also, you need the following packages in your python virtual environment:

- OpenCV
- InfluxDB 

# How to Run
```python source_watcher.py```

The program will ask you for the following inputs:

- Tunnel or local
- Camera username and password
- Camera IP

- (if tunnel) local forward port

Note that this code will automatically generate a URL to access to your camera. To close the program, simply close the camera window, press q in the camera window, or do ctrl+C in the terminal executing the code. 

