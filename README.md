# AmBe Source Watcher - Videocamera Surveillance

This code is used to implement a person identificator using a deep neural network model. It receives the feed from a camera using the RTSP stream. The purpose of the DNN is to recognize if a person shows up in scene, record the time of first appereance and the time when the person leaves the scene. It will send an alarm to the data base alerting of the precense of unauthorized people in the area close to the radioactive source. 


## Dependencies

Before doing anything, download the following files to this directory

[MobileNetSSD_deploy.prototxt](https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view?resourcekey=0-1Lpfs4EvGDeCQz12AF64hQ)

[MobileNetSSD_deploy.caffemodel](https://gist.github.com/mm-aditya/797a3e7ee041ef88cd4d9e293eaacf9f#file-mobilenetssd_deploy-prototxt)

Also, you need the following packages in your python virtual environment:

- OpenCV (Visualize camera output and run the DNN)
- InfluxDB (Push alarms to the data base)

## How to Run
```python source_watcher.py```

The program will ask you for the following inputs:

- Tunnel or local
- Camera username and password
- Camera IP

Note that this code will automatically generate a URL to access to your camera. To close the program, simply close the camera window, press q in the camera window, or do ctrl+C in the terminal executing the code. 

## Using an SSH Tunnel

Open the SSH tunnel before running this script. Also, make sure you forward port 554 (at least for AXIS cameras), which is the one that provides the RTSP stream. For instance:

```ssh -L 10554:<CAMERA IP ADDRESS>:554 <YOUR PROXY JUMP HOST>```

During the execution of the script, enter the option tunnel. The program will ask you for the local forward port (10554 in this example).

## Debug Tips

* You can check if the camera is visible by doing:
```ping <CAMERA IP ADDRESS>```.

* To keep the program executing you can use a tmux or screen session. 

