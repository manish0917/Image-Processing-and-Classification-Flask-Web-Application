from GPSPhoto import gpsphoto
# Get the data from image file and return a dictionary
data = gpsphoto.getGPSData('/path/to/image.jpg')
rawData = gpsphoto.getRawData('/path/to/image.jpg')

# Print out just GPS Data of interest
for tag in data.keys():
    print "%s: %s" % (tag, data[tag])

# Print out raw GPS Data for debugging
for tag in rawData.keys():
    print "%s: %s" % (tag, rawData[tag])

# Create a GPSPhoto Object
photo = gpsphoto.GPSPhoto()
photo = gpsphoto.GPSPhoto("/path/to/photo.jpg")

# Create GPSInfo Data Object
info = gpsphoto.GPSInfo((35.104860, -106.628915))
info = gpsphoto.GPSInfo((35.104860, -106.628915), \
          timeStamp='1970:01:01 09:05:05')
info = gpsphoto.GPSInfo((35.104860, -106.628915), \
          alt=10, timeStamp='1970:01:01 09:05:05')

# Modify GPS Data
photo.modGPSData(info, '/path/to/newFile.jpg')

# Strip GPS Data
photo.stripData('/path/to/newFile.jpg')
