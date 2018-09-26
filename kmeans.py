import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import math

class Centroid:
	def __init__(self, lowestX, lowestY, highestX, highestY):
		self.coordinate = (randint(lowestX, highestX), randint(lowestY, highestY))
		self.followers = []
		self.colour = ""

	def addFollower(self, follower):
		self.followers.append(follower)

	def resetFollowers(self):
		self.followers = []

	def assignColour(self, colourName):
		self.colour = colourName

	def updateCoordinates(self, newCoordinate):
		self.coordinate = newCoordinate

def main():
	k = 2
	iterations = 20
	nDataPoints = 150

	lowestX = 0
	lowestY = 0
	highestX = 100
	highestY = 100

	dataPoints, lowestX, lowestY, highestX, highestY = readData('kmeans')
	# dataPoints = generateData(nDataPoints, lowestX, lowestY, highestX, highestY)
	print(f"Datapoints = {dataPoints}")
	kMeans(k, iterations, lowestX, lowestY, highestX, highestY, dataPoints)

def generateData(iterations, lowestX, lowestY, highestX, highestY):
	dataPoints = []
	for i in range(0, iterations):
		dataPoints.append((randint(lowestX, highestX), randint(lowestY, highestY)))
	return dataPoints

def readData(filename):
	results = pd.read_excel(filename + '.xlsx')
	dataPoints = []	

	lowestX = 1000000
	lowestY = 1000000
	highestX = 0
	highestY = 0
	for i in results.index:
		currentRow = results.ix[i]
		dataPoints.append((currentRow[0], currentRow[1]))
		if currentRow[0] < lowestX:
			lowestX = currentRow[0]
		if currentRow[1] < lowestY:
			lowestY = currentRow[1]
		if currentRow[0] > highestX:
			highestX = currentRow[0]
		if currentRow[1] > highestY:
			highestY = currentRow[1]
	return dataPoints, lowestX, lowestY, highestX, highestY

def kMeans(k, iterations, lowestX, lowestY, highestX, highestY, dataPoints):
	centroids = []
	for i in range(0, k):
		centroid = Centroid(lowestX, lowestY, highestX, highestY)
		centroids.append(centroid)
	
	for i in range(0, iterations):
		assignFollowers(dataPoints, centroids)

		for centroid in centroids:
			print(f"Centroid {centroid.coordinate} has followers {centroid.followers}")
		
		if i == (iterations - 1):
			scatterPlot(dataPoints, centroids)
		averageDistance(centroids)
		
def scatterPlot(dataPoints, centroids):
	colors = ['b', 'g', 'r', 'c', 'm', 'y']

	for centroid in centroids:
		x = []
		y = []
		for point in centroid.followers:
			x = np.append(x, point[0])
			y = np.append(y, point[1])

		if centroid.colour == "":
			index = randint(0, len(colors) - 1)
			centroid.assignColour(colors[index])
			colors.pop(index)
		plt.scatter(x, y, s = 10, c = centroid.colour, linewidth = 9, alpha = 0.6) # Plot all data points
		plt.scatter([centroid.coordinate[0]], [centroid.coordinate[1]], s = 15, linewidth = 10, marker = 'o', c = 'k', alpha = 1) # Plot centroid points themselves

	plt.show()

def assignFollowers(dataPoints, centroids): #Assign every data point to the closest (new) centroid
	for centroid in centroids:
		centroid.resetFollowers() # Reset the current followers from every centroid

	for point in dataPoints:
		closestCentroid = ()
		closestDistance = 1000000
		for centroid in centroids:
			currentDistance = distance(point, centroid.coordinate)
			if currentDistance <= closestDistance:
				closestDistance = currentDistance
				closestCentroid = centroid
		closestCentroid.addFollower(point)

def distance(coordinate1, coordinate2):
	xdiff = math.fabs(coordinate1[0] - coordinate2[0])
	ydiff = math.fabs(coordinate1[1] - coordinate2[1])
	distance = math.sqrt((xdiff ** 2) + (ydiff ** 2))
	return distance

def averageDistance(centroids):
	for centroid in centroids:
		xsum = 0
		ysum = 0

		if len(centroid.followers) >= 1:
			for follower in centroid.followers:
				xsum += follower[0]
				ysum += follower[1]

			averagePoint = (xsum / len(centroid.followers), ysum / len(centroid.followers))
			print(f"Centroid updated from {centroid.coordinate} to ")
			centroid.updateCoordinates(averagePoint)
			print(f"{centroid.coordinate}")


if __name__ == '__main__':
  	main()


