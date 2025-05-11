import numpy as np
import pickle as pkl
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

image_width = 640
image_height = 480
obj_tie_break = 0

files = ["30cm.pkl", "50cm.pkl", "60cm.pkl", "71cm.pkl", "90cm.pkl"]
distances = [30, 50, 60, 71, 90]
bounds = []
for file in files:
    with (open(file, "rb")) as first:
        while True:
            try:
                bounds.append(pkl.load(first))
            except EOFError:
                break

def compute_height(y_coord, obj_height,image_height):
    return image_height - (y_coord + obj_height)

y_coords = []
obj_heights = []
for bound in bounds:
    if len(bound['y']) > 1:
        print ("warning: multiple bounds in file")
    y_coords.append(bound['y'][obj_tie_break])
    obj_heights.append(bound['h'][obj_tie_break])

heights = [compute_height(y_coord, obj_height, image_height) for y_coord, obj_height in zip(y_coords, obj_heights)] # height from bottom of image in pixels
dists_from_horizon = []
for height in heights:
    dist_from_horizon = image_height//2-height
    if dist_from_horizon <= 0:
        dists_from_horizon.append(float("inf"))
    else:
        dists_from_horizon.append(dist_from_horizon)

print(y_coords)
# print([y_coord+obj_height for y_coord, obj_height in zip(y_coords, obj_heights)])
print(heights)
print(dists_from_horizon)
print(distances)

def model(x,a,b):
    return a/x + b

# params, _ = curve_fit(model, dists_from_horizon, distances)
test_id = [2]
train_id = [0,1,3,4]
# train_coords = [y_coords[i] for i in train_id]
train_coords = [dists_from_horizon[i] for i in train_id]
train_distances = [distances[i] for i in train_id]
params, _ = curve_fit(model, train_coords, train_distances)
a,b = params
predicted_dists = []
for i in test_id:
    # predcited_dists.append(model(y_coords[i], a,b))
    predicted_dists.append(model(dists_from_horizon[i], a,b))

print(predicted_dists)
fig, ax = plt.subplots()
for i, dist in zip(test_id,predicted_dists):
    x1 = bounds[i]['x'][obj_tie_break]
    x2 = x1 + bounds[i]['w'][obj_tie_break]
    y = dist
    plt.plot([x1,x2],[y,y], marker = 'o', color = 'red')

x_min = 0
x_max = image_width
ax.set_xlim(x_min, x_max)
y_min = 0
y_max = image_height//2
ax.set_ylim(y_min, y_max)

plt.show()
for i,j in enumerate(test_id):
    print(f"predicted depth: {predicted_dists[i]}")
    print(f"actual depth: {distances[j]}")