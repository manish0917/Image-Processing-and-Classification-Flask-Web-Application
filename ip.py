import requests

def getDetails():
	infdic={}	
	res = requests.get('https://ipinfo.io/')
	data = res.json()
	#print(data)
	#ip = data['ip']
	#city = data['city']
	#region = data['region']
 	#country = data['country']
 	#postal=data['postal']
	#location = data['loc'].split(',')
	#latitude = location[0]
	#longitude = location[1]

	#print("Latitude : ", latitude)
	#print("Longitude : ", longitude)
	#print("City : ", city)
	#print("Ip : ", ip)
	#print('region : ', region )

	return data
