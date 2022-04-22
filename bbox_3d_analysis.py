from bbox import BBox3D
import pyquaternion
import numpy as np
import json
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pytorch3d.ops import box3d_overlap
import torch
# Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)



def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

pred_dir = "/home/rise/Desktop/test/labels/ensenso_stacked_boxes_2700mm.json"
gt_dir = "/home/rise/Desktop/labels/ensenso_stacked_boxes_2700mm.json"

# true object associations:
# (gt id, pred id)
obj_ass =[(0,38),(1,46),(2,37),(3,33),(19,15),(4,11),(5,8),(6,17),(7,20),
            (8,10),(20,1),(9,19),(11,14),(12,16),(13,18),(14,3),(50,26),
            (21,5),(15,9),(16,2),(17,13),(18,7),(43,56),(45,34),(44,42),
            (22,0),(32,32),(33,29),(34,40),(37,36),(35,51),(36,49),(29,22),
            (27,4),(30,27),(28,12),(55,41),(40,24),(41,28),(44,42),(42,35),
            (25,23),(31,30),(26,21),(46,44),(47,43),(24,6),(23,25)]

pred_file = open(pred_dir, 'r')
preds_json = json.load(pred_file)
pred_file.close()
gt_file = open(gt_dir, 'r')
gt_json = json.load(gt_file)
gt_file.close()

print(preds_json.keys())
print(gt_json.keys())

preds_bboxes = []
for bbox_data in preds_json['objects']:
    cent = bbox_data['centroid']
    dims = bbox_data['dimensions']
    rot = bbox_data['rotations']
    bboxs = BBox3D(x=cent['x'], y=cent['y'], z=cent['z'], 
                   rw=rot['w'], rx=rot['x'], ry=rot['y'], rz = rot['z'],
                   #euler_angles=[rot['x'],rot['y'],rot['z']],
                   length = dims['length'], width = dims['width'], 
                   height = dims['height'], is_center = True) 
    preds_bboxes.append(bboxs)

gt_bboxes = []
for bbox_data in gt_json['objects']:
    cent = bbox_data['centroid']
    dims = bbox_data['dimensions']
    rot = bbox_data['rotations']
    bboxs = BBox3D(x=cent['x'], y=cent['y'], z=cent['z'], 
                   euler_angles=[rot['x'],rot['y'],rot['z']],
                   length = dims['length'], width = dims['width'], 
                   height = dims['height'], is_center = True) 
    gt_bboxes.append(bboxs)

print("Total predicted bboxes: ", len(preds_bboxes))
print("Total true boxes: ", len(gt_bboxes))
#print(gt_bboxes[0].p)
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1,projection="3d")
ax2 = fig.add_subplot(1,2,2,projection="3d")


# draw bounding boxes
colors = plt.cm.plasma(np.arange(0,len(preds_bboxes))/len(preds_bboxes))
for i, box in enumerate(preds_bboxes[:]):
    if(np.max(box.p)>10):
        continue
    cube = box.p

    # plot vertices
    ax1.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], s = 0.3)

    # write corners' names
    texts = ["p"+str(i+1) for i in range(8)]
    corners = [box.p1, box.p2, box.p3, box.p4, box.p5, box.p6, box.p7, box.p8]
    corners.sort(key=(lambda x: x[0]))
    for k,(text, corner) in enumerate(zip(texts,corners)):
        continue
        if k < 4:
            ax1.text(corner[0],corner[1], corner[2], text, color = "k")

    # list of sides' polygons of figure
    verts = [[cube[0],cube[1],cube[2],cube[3]],
            [cube[4],cube[5],cube[6],cube[7]], 
            [cube[0],cube[1],cube[5],cube[4]], 
            [cube[2],cube[3],cube[7],cube[6]], 
            [cube[1],cube[2],cube[6],cube[5]],
            [cube[4],cube[7],cube[3],cube[0]]]

    # plot sides
    ax1.add_collection3d(Poly3DCollection(verts,  facecolors=colors[i], linewidths=1, edgecolors='b', alpha=.25))
    ax1.text(box.center[0], box.center[1], box.center[2], str(i))


colors = plt.cm.plasma(np.arange(0,len(gt_bboxes))/len(gt_bboxes))
for i, box in enumerate(gt_bboxes[:]):
    if(np.max(box.p)>10):
        continue
    cube = box.p

    # plot vertices
    ax2.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], s=0.3)

    # write corners' names
    texts = ["p"+str(i+1) for i in range(8)]
    corners = [box.p1, box.p2, box.p3, box.p4, box.p5, box.p6, box.p7, box.p8]
    corners.sort(key=(lambda x: x[0]))
    for k,(text, corner) in enumerate(zip(texts,corners)):
        continue
        if k < 4:
            ax2.text(corner[0],corner[1], corner[2], text, color = "k")

    # list of sides' polygons of figure
    verts = [[cube[0],cube[1],cube[2],cube[3]],
            [cube[4],cube[5],cube[6],cube[7]], 
            [cube[0],cube[1],cube[5],cube[4]], 
            [cube[2],cube[3],cube[7],cube[6]], 
            [cube[1],cube[2],cube[6],cube[5]],
            [cube[4],cube[7],cube[3],cube[0]]]

    # plot sides
    ax2.add_collection3d(Poly3DCollection(verts,  facecolors=colors[i], linewidths=1, edgecolors='b', alpha=.25))
    ax2.text(box.center[0], box.center[1], box.center[2], str(i))

fig2 = plt.figure()
ax3 = fig2.add_subplot(1,1,1,projection="3d")

gt_colors = plt.cm.viridis(np.arange(0,len(obj_ass))/len(obj_ass))
pred_colors = plt.cm.copper(np.arange(0,len(obj_ass))/len(obj_ass))
for i, index_pair in enumerate(obj_ass[:]):
    # retrieve corresponding boxes
    gt_box = gt_bboxes[index_pair[0]]
    pred_box = preds_bboxes[index_pair[1]]

    gt_cube = gt_box.p
    pred_cube = pred_box.p
    # plot vertices
    ax3.scatter3D(gt_cube[:, 0], gt_cube[:, 1], gt_cube[:, 2], s=0.3, color=gt_colors[i])
    ax3.scatter3D(pred_cube[:, 0], pred_cube[:, 1], pred_cube[:, 2], s=0.3, color=pred_colors[i])

    # write corners' names
    texts = ["p"+str(i+1) for i in range(8)]
    corners = [box.p1, box.p2, box.p3, box.p4, box.p5, box.p6, box.p7, box.p8]
    corners.sort(key=(lambda x: x[0]))
    for k,(text, corner) in enumerate(zip(texts,corners)):
        continue
        if k < 4:
            ax2.text(corner[0],corner[1], corner[2], text, color = "k")

    # list of sides' polygons of ground true cube
    gt_verts = [[gt_cube[0],gt_cube[1],gt_cube[2],gt_cube[3]],
            [gt_cube[4],gt_cube[5],gt_cube[6],gt_cube[7]], 
            [gt_cube[0],gt_cube[1],gt_cube[5],gt_cube[4]], 
            [gt_cube[2],gt_cube[3],gt_cube[7],gt_cube[6]], 
            [gt_cube[1],gt_cube[2],gt_cube[6],gt_cube[5]],
            [gt_cube[4],gt_cube[7],gt_cube[3],gt_cube[0]]]

    # plot sides
    ax3.add_collection3d(Poly3DCollection(gt_verts,  facecolors=gt_colors[i], linewidths=1, edgecolors=gt_colors[i], alpha=.25))
    ax3.text(box.center[0], box.center[1], box.center[2], str(index_pair[0]), color= gt_colors[i])

    # list of sides' polygons of predicted cube
    pred_verts = [[pred_cube[0],pred_cube[1],pred_cube[2],pred_cube[3]],
            [pred_cube[4],pred_cube[5],pred_cube[6],pred_cube[7]], 
            [pred_cube[0],pred_cube[1],pred_cube[5],pred_cube[4]], 
            [pred_cube[2],pred_cube[3],pred_cube[7],pred_cube[6]], 
            [pred_cube[1],pred_cube[2],pred_cube[6],pred_cube[5]],
            [pred_cube[4],pred_cube[7],pred_cube[3],pred_cube[0]]]

    # plot sides
    ax3.add_collection3d(Poly3DCollection(pred_verts,  facecolors=pred_colors[i], linewidths=1, edgecolors=pred_colors[i], alpha=.25))
    ax3.text(box.center[0], box.center[1], box.center[2], str(index_pair[1]), color= pred_colors[i])

    intersection_vol, iou_3d = box3d_overal(torch.tensor(gt_cube), torch.tensor(pred_cube))
    print(i,intersection_vol,iou_3d)

ax1.set_title("Predicted Positions")
ax2.set_title("Ground Truth Positions")

ax1.set_box_aspect([1,1,1])
ax2.set_box_aspect([1,1,1])
ax3.set_box_aspect([1,1,1])

# draw coordinate system
ax1.quiver(0,0,0,1,0,0, color = "r", arrow_length_ratio = 0.1)
ax1.quiver(0,0,0,0,1,0, color = "g", arrow_length_ratio = 0.1)
ax1.quiver(0,0,0,0,0,1, color = "b", arrow_length_ratio = 0.1)
ax1.text(0,0,0,"{Robot Base Frame}",color="k")

ax2.quiver(0,0,0,1,0,0, color = "r", arrow_length_ratio = 0.1)
ax2.quiver(0,0,0,0,1,0, color = "g", arrow_length_ratio = 0.1)
ax2.quiver(0,0,0,0,0,1, color = "b", arrow_length_ratio = 0.1)
ax2.text(0,0,0,"{Robot Base Frame}",color="k")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

set_axes_equal(ax1)
set_axes_equal(ax2)
set_axes_equal(ax3)

plt.tight_layout()
plt.show()
    

