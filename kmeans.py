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
	k = 3
	iterations = 50
	nDataPoints = 750

	lowestX = 0
	lowestY = 0
	highestX = 500
	highestY = 500

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


# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def load_dataset(name):
#     return np.loadtxt(name)


# def euclidian(a, b):
#     return np.linalg.norm(a-b)


# def plot(dataset, history_centroids, belongs_to):
#     colors = ['r', 'g']

#     fig, ax = plt.subplots()

#     for index in range(dataset.shape[0]):
#         instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
#         for instance_index in instances_close:
#             ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))

#     history_points = []
#     for index, centroids in enumerate(history_centroids):
#         for inner, item in enumerate(centroids):
#             if index == 0:
#                 history_points.append(ax.plot(item[0], item[1], 'bo')[0])
#             else:
#                 history_points[inner].set_data(item[0], item[1])
#                 print("centroids {} {}".format(index, item))

#                 plt.pause(0.8)


# def kmeans(k, epsilon=0, distance='euclidian'):
#     history_centroids = []
#     if distance == 'euclidian':
#         dist_method = euclidian
#     dataset = load_dataset('durudataset.txt')
#     # dataset = dataset[:, 0:dataset.shape[1] - 1]
#     num_instances, num_features = dataset.shape
#     prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
#     history_centroids.append(prototypes)
#     prototypes_old = np.zeros(prototypes.shape)
#     belongs_to = np.zeros((num_instances, 1))
#     norm = dist_method(prototypes, prototypes_old)
#     iteration = 0
#     while norm > epsilon:
#         iteration += 1
#         norm = dist_method(prototypes, prototypes_old)
#         prototypes_old = prototypes
#         for index_instance, instance in enumerate(dataset):
#             dist_vec = np.zeros((k, 1))
#             for index_prototype, prototype in enumerate(prototypes):
#                 dist_vec[index_prototype] = dist_method(prototype,
#                                                         instance)

#             belongs_to[index_instance, 0] = np.argmin(dist_vec)

#         tmp_prototypes = np.zeros((k, num_features))

#         for index in range(len(prototypes)):
#             instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
#             prototype = np.mean(dataset[instances_close], axis=0)
#             # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
#             tmp_prototypes[index, :] = prototype

#         prototypes = tmp_prototypes

#         history_centroids.append(tmp_prototypes)

#     # plot(dataset, history_centroids, belongs_to)

#     return prototypes, history_centroids, belongs_to


# def execute():
#     dataset = load_dataset('durudataset.txt')
#     centroids, history_centroids, belongs_to = kmeans(2)
#     plot(dataset, history_centroids, belongs_to)

# execute()