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

pred_dir = "/home/rise/Desktop/test/labels/azure_stacked_boxes_2400mm.json"
gt_dir = "/home/rise/Desktop/labels/azure_stacked_boxes_2400mm.json"

# true object associations:
# (gt id, pred id)
obj_ass =[(0,38),(1,46),(2,37),(3,33),(19,15),(4,11),(5,8),(6,17),(7,20),
            (8,10),(20,1),(9,19),(11,14),(12,16),(13,18),(14,3),(50,26),
            (21,5),(15,9),(16,2),(17,13),(18,7),(43,56),(45,34), (48,31), (49,39),
            (22,0),(32,32),(33,29),(34,40),(37,36),(35,51),(36,49),(29,22),
            (27,4),(30,27),(28,12),(55,41),(40,24),(41,28),(44,42),(42,35),
            (25,23),(31,30),(26,21),(46,44),(47,43),(24,6),(23,25)]

obj_ass.sort(key=(lambda x: x[0]))

pred_file = open(pred_dir, 'r')
preds_json = json.load(pred_file)
pred_file.close()
gt_file = open(gt_dir, 'r')
gt_json = json.load(gt_file)
gt_file.close()

#print(preds_json.keys())
#print(gt_json.keys())

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
ax3 = fig2.add_subplot(1,2,1,projection="3d")
ax4 = fig2.add_subplot(1,2,2,projection="3d")

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
    #ax3.text(box.center[0], box.center[1], box.center[2], str(index_pair[1]), color= pred_colors[i])


# compute IoU to get prediction-ground truth pairs
pred_idxs = np.arange(len(preds_bboxes)).tolist()
gt_idxs = np.arange(len(gt_bboxes)).tolist()
ass_found = []   # store object associations found
ious = []
int_vol = []
for k, pred_idx in enumerate(pred_idxs):
    pred_box = torch.tensor(preds_bboxes[k].p, dtype=torch.float32)
    max_iou = -1
    max_gt_index = None
    for j, gt_idx in enumerate(gt_idxs):
        if gt_idx == -1: # ignore if this gt_idx was already used
            continue
        gt_box = torch.tensor(gt_bboxes[j].p, dtype=torch.float32)
        intersection_vol, iou_3d = box3d_overlap(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
        ious.append(iou_3d.item())
        int_vol.append(intersection_vol)
        if (iou_3d > 0.1):   # only consider match if iou > 0.5
            if (iou_3d > max_iou):
                max_iou = iou_3d
                max_gt_index = gt_idx
    
    if max_gt_index is None:
        continue  # match not found
    else:
        ass_found.append((max_gt_index,pred_idx))
        gt_idxs[max_gt_index] = -1

ass_found.sort(key=(lambda x: x[0]))
#print("True associations: ",len(obj_ass))
#print(obj_ass)
#print("Found associations: ",len(ass_found))
#print(ass_found)
correct_ass = 0
for ass in ass_found:
    if ass in obj_ass:
        correct_ass +=1
    else:
        print("Incorrect association: ",ass)
print("Correct associations: ", correct_ass)
print("Ious: min {} max {} std {} mean {}".format(np.min(ious), np.max(ious),np.std(ious),np.mean(ious)))
print("Int Volume: min {} max {} std {} mean {}".format(np.min(int_vol), np.max(int_vol),np.std(int_vol),np.mean(int_vol)))

gt_colors = plt.cm.viridis(np.arange(0,len(ass_found))/len(ass_found))
pred_colors = plt.cm.copper(np.arange(0,len(ass_found))/len(ass_found))
for i, index_pair in enumerate(ass_found[:]):
    # retrieve corresponding boxes
    gt_box = gt_bboxes[index_pair[0]]
    pred_box = preds_bboxes[index_pair[1]]

    gt_cube = gt_box.p
    pred_cube = pred_box.p
    # plot vertices
    ax4.scatter3D(gt_cube[:, 0], gt_cube[:, 1], gt_cube[:, 2], s=0.3, color=gt_colors[i])
    ax4.scatter3D(pred_cube[:, 0], pred_cube[:, 1], pred_cube[:, 2], s=0.3, color=pred_colors[i])

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
    ax4.add_collection3d(Poly3DCollection(gt_verts,  facecolors=gt_colors[i], linewidths=1, edgecolors=gt_colors[i], alpha=.25))
    #ax4.text(box.center[0], box.center[1], box.center[2], str(index_pair[0]), color= gt_colors[i])

    # list of sides' polygons of predicted cube
    pred_verts = [[pred_cube[0],pred_cube[1],pred_cube[2],pred_cube[3]],
            [pred_cube[4],pred_cube[5],pred_cube[6],pred_cube[7]], 
            [pred_cube[0],pred_cube[1],pred_cube[5],pred_cube[4]], 
            [pred_cube[2],pred_cube[3],pred_cube[7],pred_cube[6]], 
            [pred_cube[1],pred_cube[2],pred_cube[6],pred_cube[5]],
            [pred_cube[4],pred_cube[7],pred_cube[3],pred_cube[0]]]

    # plot sides
    ax4.add_collection3d(Poly3DCollection(pred_verts,  facecolors=pred_colors[i], linewidths=1, edgecolors=pred_colors[i], alpha=.25))
    #ax4.text(box.center[0], box.center[1], box.center[2], str(index_pair[1]), color= pred_colors[i])        

# get pairs of points
pred_boxes_corner_idx = []   # store idx of corners for each predicted box
gt_boxes_corner_idx = []     # store idx of corner in each gt box that corresponds to each corner in pred box.. together with distnce
pred_colors = plt.cm.prism(np.arange(0,len(ass_found))/len(ass_found))
gt_colors = plt.cm.gist_ncar(np.arange(0,len(ass_found))/len(ass_found))

for k, index_pair in enumerate(ass_found[:]):
#for k, index_pair in enumerate(random.sample(ass_found[:],2)):

    # get only the 4 points with lowest X position value : front plane points
    # retrieve corresponding boxes
    gt_box = gt_bboxes[index_pair[0]]
    pred_box = preds_bboxes[index_pair[1]]

    # get corners of 3d box and order by x value
    gt_cube = gt_box.p.tolist()
    pred_cube = pred_box.p.tolist()    
    gt_cube.sort(key=(lambda x: x[0]))
    pred_cube.sort(key=(lambda x: x[0]))

    # get only front plane corners
    gt_front_corners = gt_cube[:4]
    pred_front_corners = pred_cube[:4]
    #print("GT front conrners",gt_front_corners)
    #print("Pred front corners",pred_front_corners)
    
    # for each corner in front plane, get the index of the 
    # gt corner with smallest euclidean distance -> correspondance search
    # ... I hope the mapping is biyective
    pred_box_corner_ids = []
    gt_box_corner_ids = []
    for n, pred_corner in enumerate(pred_front_corners):
        pred_box_corner_ids.append(n)
        idx_dist = []
        for j, gt_corner in enumerate(gt_front_corners):
            
            idx_dist.append((j,np.linalg.norm(np.array(pred_corner) - np.array(gt_corner))))

        # sort distances
        idx_dist.sort(key=(lambda x: x[1]))
        #print("Distance sort\n", idx_dist)

        gt_box_corner_ids.append(idx_dist[0])

        ax4.text(pred_corner[0], pred_corner[1], pred_corner[2], str(k*4 + n), color= pred_colors[k])
        ax4.text(gt_front_corners[idx_dist[0][0]][0], gt_front_corners[idx_dist[0][0]][1], gt_front_corners[idx_dist[0][0]][2], str(k*4 + n), color= gt_colors[k])
    
    pred_boxes_corner_idx.append(pred_box_corner_ids)
    gt_boxes_corner_idx.append(gt_box_corner_ids)

#print("Pred Corners",(pred_boxes_corner_idx))
#print("GT Corners",(gt_boxes_corner_idx))

# sum all per corner errors into ADD metric
add = []
for data in  gt_boxes_corner_idx:
    error = 0
    for corner in data:
        error += corner[1]
    add.append(error/4)

assert len(add)==len(gt_boxes_corner_idx)
print("ADD: n= {} min {} max {} std {} mean {}".format(len(add),np.min(add), np.max(add),np.std(add),np.mean(add)))

fig3 = plt.figure()
ax5 = fig3.add_subplot(1,1,1)
ax5.hist(add,bins=10,label="ADD_4 error")
ax5.set_title("Azure Kinect DK ADD4 error \n Stacked Boxes: distance = 2400mm  n = {}".format(len(add)))
ax5.set_xlabel("ADD error")
ax5.set_ylabel("Sample count")

ax1.set_title("Predicted Positions")
ax2.set_title("Ground Truth Positions")
ax3.set_title("True Associations")
ax4.set_title("Founds Associations")


ax1.set_box_aspect([1,1,1])
ax2.set_box_aspect([1,1,1])
ax3.set_box_aspect([1,1,1])
ax4.set_box_aspect([1,1,1])

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
set_axes_equal(ax4)

plt.tight_layout()
plt.show()
    

