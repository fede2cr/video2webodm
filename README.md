# video2webodm

WebODM normally uses photos from a drone that have GPS encoded in the EXIF of each image.

This set of script manage to:
- [ ] Grab a video from a DJI drone
- [ ] Split the video in images so that the overlap in between is what WebODM requires for stitching
- [ ] Takes the GPS data out of the subtitle encoded in the video
- [ ] Adds the GPS data to the EXIF of the extracted frame

How much time passes in between each extracted frame is auto estimated, as it is a variable parameter that depends on the distance from the drone camera to the terrain, and the speed of the drone.

Now you can import the images to WebODM without having to manual define feature points.


## Usage

```
./v2wodm.py -i input.mp4 -start seconds -overlap pct -end seconds

input.mp4: the name of the video file to process
seconds: start and finish times to process. In case the recording started befor the terrain of interest comes into view.
overlap: WebODM requires a certain overlap, measure in percent
```