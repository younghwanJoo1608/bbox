from bbox import BBox3D
import pyquaternion
import numpy as np
import numpy.linalg as la
import json
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from pytorch3d.ops import box3d_overlap
import torch
import math
# Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)

_box_triangles = [
    [0, 1, 2],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [1, 5, 6],
    [1, 6, 2],
    [0, 4, 7],
    [0, 7, 3],
    [3, 2, 6],
    [3, 6, 7],
    [0, 1, 5],
    [0, 4, 5],
]

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

# pred_dir = "/home/jyh/Documents/obb_error/sipmask_convnext_albu_aug_1024/azure_1900/5.json"
# gt_dir = "/home/jyh/Documents/obb_error/evaluation/azure_scramble_1900mm.json"

# pred_dir = "/home/jyh/Documents/obb_error/sipmask_convnext_albu_aug_1024/azure_2400/1.json"
# gt_dir = "/home/jyh/Documents/obb_error/evaluation/azure_stacked_boxes_2400mm.json"

# pred_dir = "/home/jyh/Documents/obb_error/sipmask_convnext_albu_aug_1024/azure_1670/4.json"
# gt_dir = "/home/jyh/Documents/obb_error/evaluation/azure_pouch_box_1670mm.json"

pred_dir = "/home/jyh/Documents/obb_error/sipmask_convnext_albu_aug_1024/2m/5.json"
gt_dir = "/home/jyh/Documents/obb_error/evaluation/2m.json"

# true object associations:
# (gt id, pred id)

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


# draw  predicted bounding boxes
colors = plt.cm.plasma(np.arange(0,len(preds_bboxes))/len(preds_bboxes))
for i, box in enumerate(preds_bboxes[:]):
    if(np.max(box.p)>4):
        continue
    cube = box.p

    # plot vertices
    ax1.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], s = 0.3)

    # write corners' names
    texts = ["p"+str(i+1) for i in range(8)]
    corners = [box.p1, box.p2, box.p3, box.p4, box.p5, box.p6, box.p7, box.p8]
    #corners.sort(key=(lambda x: x[0]))
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
    ax1.text(corners[1][0],corners[1][1],corners[1][2],"p2",color=colors[i])
    ax1.text(corners[2][0],corners[2][1],corners[2][2],"p3",color=colors[i])
    #ax1.text(box.center[0], box.center[1], box.center[2], str(i))

# draw ground truth bounding boxes
colors = plt.cm.plasma(np.arange(0,len(gt_bboxes))/len(gt_bboxes))
for i, box in enumerate(gt_bboxes[:]):
    if(np.max(box.p)>4):
        continue
    cube = box.p

    # plot vertices
    ax2.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], s=0.3)

    # write corners' names
    texts = ["p"+str(i+1) for i in range(8)]
    corners = [box.p1, box.p2, box.p3, box.p4, box.p5, box.p6, box.p7, box.p8]
    #corners.sort(key=(lambda x: x[0]))
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
    ax2.text(corners[1][0],corners[1][1],corners[1][2],"p2",color=colors[i])
    ax2.text(corners[2][0],corners[2][1],corners[2][2],"p3",color=colors[i])
    #ax2.text(box.center[0], box.center[1], box.center[2], str(i))


fig2 = plt.figure()
# ax3 = fig2.add_subplot(1,2,1,projection="3d")
# ax4 = fig2.add_subplot(1,2,2,projection="3d")
ax4 = fig2.add_subplot(1,1,1,projection="3d")

# draw ground true associations... do this only if these are known
# usually these are unknown
gt_colors = plt.cm.viridis(np.arange(0,len(gt_bboxes))/len(gt_bboxes))
pred_colors = plt.cm.copper(np.arange(0,len(preds_bboxes))/len(preds_bboxes))


# compute IoU to get association between prediction & ground truth 
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
        #if pred_idx == 31:
        
        faces = torch.tensor(_box_triangles, dtype=torch.int64, device=pred_box.unsqueeze(0).device)
        verts = pred_box.unsqueeze(0).index_select(index=faces.view(-1), dim=1)
        B = pred_box.unsqueeze(0).shape[0]
        T, V = faces.shape
        # (B, T, 3, 3) -> (B, T, 3)
        v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

        normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
        face_areas = normals.norm(dim=-1) / 2

        if (face_areas < 1e-4).any().item():
            continue
    
        intersection_vol, iou_3d = box3d_overlap(pred_box.unsqueeze(0), gt_box.unsqueeze(0))
        ious.append(iou_3d.item())
        int_vol.append(intersection_vol)
        if (iou_3d > 0.1):   # only consider match if iou > threshold
            if (iou_3d > max_iou):
                max_iou = iou_3d
                max_gt_index = gt_idx
    
    if max_gt_index is None:
        continue  # match not found
    else:
        ass_found.append((max_gt_index,pred_idx))
        gt_idxs[max_gt_index] = -1

ass_found.sort(key=(lambda x: x[0]))  #sort based on gt index

#print("True associations: ",len(obj_ass))
#print(obj_ass)
#print("Found associations: ",len(ass_found))
#print(ass_found)
#correct_ass = 0
#for ass in ass_found:
#    if ass in obj_ass:
#        correct_ass +=1
#    else:
#        print("Incorrect association: ",ass)
#print("Correct associations: ", correct_ass)
#print("Ious: min {} max {} std {} mean {}".format(np.min(ious), np.max(ious),np.std(ious),np.mean(ious)))
#print("Int Volume: min {} max {} std {} mean {}".format(np.min(int_vol), np.max(int_vol),np.std(int_vol),np.mean(int_vol)))


# draw found correspondences in the same coordinate system
# this is just for visualization purposes
gt_colors = plt.cm.viridis(np.arange(0,len(ass_found))/len(ass_found))
pred_colors = plt.cm.flag(np.arange(0,len(ass_found))/len(ass_found))
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
z_errors = []

# Find prediction - gt correspondence of corners in the front plane
# For each pbox in pred_box
#   For each pcorner in pbox:
#       for each gtcorner in gt_box
#              Get euclidian distance between pcorner and gtcorner
#   sort euclidean distance from shortest to largest
#   correspondance is the gtcorner with shortest distance to pcorner
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
        # compute distance from this pred_corner to all gt_corners
        for j, gt_corner in enumerate(gt_front_corners):
            idx_dist.append((j,np.linalg.norm(np.array(pred_corner) - np.array(gt_corner)), np.abs(pred_corner[2] - gt_corner[2])))

        # sort distances from nearest to furthest
        idx_dist.sort(key=(lambda x: x[1]))
        #print("Distance sort\n", idx_dist)

        # corresponding corner should bethe one with shortest distance
        gt_box_corner_ids.append(idx_dist[0])
        z_errors.append(idx_dist[0][2])

        #ax4.text(pred_corner[0], pred_corner[1], pred_corner[2], str(k*4 + n), color= pred_colors[k])
        #ax4.text(gt_front_corners[idx_dist[0][0]][0], gt_front_corners[idx_dist[0][0]][1], gt_front_corners[idx_dist[0][0]][2], str(k*4 + n), color= gt_colors[k])
    
    pred_boxes_corner_idx.append(pred_box_corner_ids)
    gt_boxes_corner_idx.append(gt_box_corner_ids)
print("z_errors: ",len(z_errors),np.mean(z_errors))
#print("Pred Corners",(pred_boxes_corner_idx))
#print("GT Corners",(gt_boxes_corner_idx))


# compute width and height
# width = shortest distance between 4 corner points
# height = 2nd shortest distnace between 4 corner points
gt_width = []
gt_heigth = []
pred_width = []
pred_heigth = []
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
    gt_front_corners = gt_cube[:3]
    pred_front_corners = pred_cube[:3]

    # compute distance between points in the front face of gt bbox
    # there are 4 points... 3 distances
    gt_dims = []
    gt_dims.append((np.linalg.norm(np.array(gt_front_corners[0])-np.array(gt_front_corners[1])),gt_front_corners[0],gt_front_corners[1]))
    gt_dims.append((np.linalg.norm(np.array(gt_front_corners[0])-np.array(gt_front_corners[2])),gt_front_corners[0],gt_front_corners[2]))
    gt_dims.append((np.linalg.norm(np.array(gt_front_corners[1])-np.array(gt_front_corners[2])),gt_front_corners[1],gt_front_corners[2]))
    gt_dims.sort(key=(lambda x: x[0]))
    #print("gtdims",gt_dims)
    gt_width.append(gt_dims[0][0]) # assume width is the smallest dimension
    gt_heigth.append(gt_dims[1][0]) # assume height is the next smallest dimension 
    
    # compute distance between points in the front face of pred box
    # again there are 4 points... 3 distances
    pred_dims = []
    pred_dims.append((np.linalg.norm(np.array(pred_front_corners[0])-np.array(pred_front_corners[1])),pred_front_corners[0],pred_front_corners[1]))
    pred_dims.append((np.linalg.norm(np.array(pred_front_corners[0])-np.array(pred_front_corners[2])),pred_front_corners[0],pred_front_corners[2]))
    pred_dims.append((np.linalg.norm(np.array(pred_front_corners[1])-np.array(pred_front_corners[2])),pred_front_corners[1],pred_front_corners[2]))
    pred_dims.sort(key=(lambda x: x[0]))
    pred_width.append(pred_dims[0][0]) # assume width is the smallest dimension
    pred_heigth.append(pred_dims[1][0]) # assume height is the next smallest dimension

    # ddraw predicted box width and height
    ax1.plot((pred_dims[0][2][0],pred_dims[0][1][0]),(pred_dims[0][2][1],pred_dims[0][1][1]),(pred_dims[0][2][2],pred_dims[0][1][2]),marker='o',color="r")
    ax1.plot((pred_dims[1][2][0],pred_dims[1][1][0]),(pred_dims[1][2][1],pred_dims[1][1][1]),(pred_dims[1][2][2],pred_dims[1][1][2]),marker='o',color="b")

    # draw gt box width and heights
    ax2.plot((gt_dims[0][2][0],gt_dims[0][1][0]),(gt_dims[0][2][1],gt_dims[0][1][1]),(gt_dims[0][2][2],gt_dims[0][1][2]),marker='o',color="r")
    ax2.plot((gt_dims[1][2][0],gt_dims[1][1][0]),(gt_dims[1][2][1],gt_dims[1][1][1]),(gt_dims[1][2][2],gt_dims[1][1][2]),marker='o',color="b")

error_width =[np.abs(v) for v in  (np.array(pred_width) - np.array(gt_width))]
error_height = [np.abs(v) for v in ( np.array(pred_heigth) - np.array(gt_heigth))]
print("Width Error:\n{}\n{}\n".format(np.mean(error_width), np.max(error_width)))
print("Hieght Error: \n{}\n{}\n".format(np.mean(error_height), np.max(error_height)))


# sum all per corner errors into ADD metric
add = []
for data in  gt_boxes_corner_idx:
    error = 0
    for corner in data:
        error += corner[1]
    add.append(error/4)

#assert len(add)==len(gt_boxes_corner_idx)
print("ADD:\n{}\n{}\n".format(np.mean(add), np.max(add)))


# compute front plane position errors
centroid_errors = []
for k, index_pair in enumerate(ass_found[:]):

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

    pred_centroid = np.mean(np.array(pred_front_corners), axis=0)
    gt_centroid = np.mean(np.array(gt_front_corners), axis=0)

    centroid_error = np.linalg.norm(gt_centroid - pred_centroid)
    centroid_errors.append(centroid_error)

    # ax4.plot(pred_centroid[0],pred_centroid[1],pred_centroid[2],marker='o',color="r")
    # ax4.plot(gt_centroid[0],gt_centroid[1],gt_centroid[2],marker='o',color="b")

print("Position Errors: \n{}\n{}\n".format(np.mean(centroid_errors), np.max(centroid_errors)))

# compute front plane orientation errors
avg_vectors = []
orientation_errors = []
for k, index_pair in enumerate(ass_found[:]):
    # get only the 4 points with lowest X position value : front plane points
    # retrieve corresponding boxes
    gt_box = gt_bboxes[index_pair[0]]
    pred_box = preds_bboxes[index_pair[1]]

    # get corners of 3d box and order by x value
    gt_cube = gt_box.p.tolist()
    pred_cube = pred_box.p.tolist()
    gt_cube.sort(key=(lambda x: x[0]))
    pred_cube.sort(key=(lambda x: (x[0], x[2])))

    # get only front plane corners
    gt_front_corners_1 = gt_cube[:4]
    gt_front_corners_2 = gt_cube[:2] + gt_cube[4:6]
    pred_front_corners_1 = pred_cube[:4]
    pred_front_corners_2 = pred_cube[:2] + pred_cube[4:6]

    # get centroid & normal vector of 1st gt plane
    gt_centroid_1 = np.mean(np.array(gt_front_corners_1), axis=0)
    gt_v1 = np.array(gt_front_corners_1[0]) - np.array(gt_front_corners_1[1])
    gt_v2 = np.array(gt_front_corners_1[0]) - np.array(gt_front_corners_1[2])
    gt_normal_v1 = np.cross(gt_v1, gt_v2)

    if gt_normal_v1[0] > 0:
        gt_normal_v1 *= -1
    u1, v1, w1 = gt_centroid_1
    x1, y1, z1 = gt_normal_v1

    # get centroid & normal vector of 2nd gt plane
    gt_centroid_2 = np.mean(np.array(gt_front_corners_2), axis=0)
    gt_v1 = np.array(gt_front_corners_2[0]) - np.array(gt_front_corners_2[1])
    gt_v2 = np.array(gt_front_corners_2[0]) - np.array(gt_front_corners_2[2])
    gt_normal_v2 = np.cross(gt_v1, gt_v2)

    if gt_normal_v2[0] > 0:
        gt_normal_v2 *= -1
    u2, v2, w2 = gt_centroid_2
    x2, y2, z2 = gt_normal_v2

    gt_normal_compare = np.array(gt_normal_v1) - np.array(gt_normal_v2)

    # If wrong front plane detected, two normal vectors are same.
    # So, change the corner of front plane and re-calculate.
    if la.norm(gt_normal_compare) < 0.001:
        gt_front_corners_1 = gt_cube[:3] + [gt_cube[4]]
        gt_front_corners_2 = gt_cube[:2] + [gt_cube[3]] + [gt_cube[5]]

        # get centroid & normal vector of 1st gt plane
        ## TODO the front plane corner may be index 0, 1, 2, 5 and 0, 1, 3, 4
        ## if so, the centroid can be different but orientation calculating is work yet.
        gt_centroid_1 = np.mean(np.array(gt_front_corners_1), axis=0)
        gt_v1 = np.array(gt_front_corners_1[0]) - np.array(gt_front_corners_1[1])
        gt_v2 = np.array(gt_front_corners_1[0]) - np.array(gt_front_corners_1[2])
        gt_normal_v1 = np.cross(gt_v1, gt_v2)

        if gt_normal_v1[0] > 0:
            gt_normal_v1 *= -1
        u1, v1, w1 = gt_centroid_1
        x1, y1, z1 = gt_normal_v1

        # get centroid & normal vector of 2nd gt plane
        gt_centroid_2 = np.mean(np.array(gt_front_corners_2), axis=0)
        gt_v1 = np.array(gt_front_corners_2[0]) - np.array(gt_front_corners_2[1])
        gt_v2 = np.array(gt_front_corners_2[0]) - np.array(gt_front_corners_2[2])
        gt_normal_v2 = np.cross(gt_v1, gt_v2)

        if gt_normal_v2[0] > 0:
            gt_normal_v2 *= -1
        u2, v2, w2 = gt_centroid_2
        x2, y2, z2 = gt_normal_v2

    # get centroid & normal vector of 1st pred plane
    pred_centroid_1 = np.mean(np.array(pred_front_corners_1), axis=0)
    x5, y5, z5 = pred_centroid_1

    pred_v1 = np.array(pred_front_corners_1[0] - np.array(pred_front_corners_1[1]))
    pred_v2 = np.array(pred_front_corners_1[0] - np.array(pred_front_corners_1[3]))
    pred_normal_v1 = np.cross(pred_v1, pred_v2)

    if pred_normal_v1[0] > 0:
        pred_normal_v1 *= -1
    x7, y7, z7 = pred_normal_v1
    # ax4.quiver(x5, y5, z5, x7, y7, z7, color="r", arrow_length_ratio=0.2, length=0.3, normalize=True)

    # get centroid & normal vector of 2nd pred plane
    pred_centroid_2 = np.mean(np.array(pred_front_corners_2), axis=0)
    x6, y6, z6 = pred_centroid_2
    # ax4.scatter(x6, y6, z6, color='b', marker='o')

    pred_v1 = np.array(pred_front_corners_2[0] - np.array(pred_front_corners_2[1]))
    pred_v2 = np.array(pred_front_corners_2[0] - np.array(pred_front_corners_2[3]))
    pred_normal_v2 = np.cross(pred_v1, pred_v2)

    if pred_normal_v2[0] > 0:
        pred_normal_v2 *= -1
    x8, y8, z8 = pred_normal_v2
    # ax4.quiver(x6, y6, z6, x8, y8, z8, color="b", arrow_length_ratio=0.2, length=0.3, normalize=True)

    pred_normal_compare = np.array(pred_normal_v1) - np.array(pred_normal_v2)

    if la.norm(pred_normal_compare) < 0.001:
        pred_front_corners_1 = pred_cube[:3] + [pred_cube[4]]
        pred_front_corners_2 = pred_cube[:2] + [pred_cube[3]] + [pred_cube[5]]

        # get centroid & normal vector of 1st gt plane
        pred_centroid_1 = np.mean(np.array(pred_front_corners_1), axis=0)
        pred_v1 = np.array(pred_front_corners_1[0]) - np.array(pred_front_corners_1[1])
        pred_v2 = np.array(pred_front_corners_1[0]) - np.array(pred_front_corners_1[2])
        pred_normal_v1 = np.cross(pred_v1, pred_v2)

        if pred_normal_v1[0] > 0:
            pred_normal_v1 *= -1
        x5, y5, z5 = pred_centroid_1
        x7, y7, z7 = pred_normal_v1

        # get centroid & normal vector of 2nd gt plane
        pred_centroid_2 = np.mean(np.array(pred_front_corners_2), axis=0)
        pred_v1 = np.array(pred_front_corners_2[0]) - np.array(pred_front_corners_2[1])
        pred_v2 = np.array(pred_front_corners_2[0]) - np.array(pred_front_corners_2[2])
        pred_normal_v2 = np.cross(pred_v1, pred_v2)

        if pred_normal_v2[0] > 0:
            pred_normal_v2 *= -1
        x6, y6, z6 = pred_centroid_2
        x8, y8, z8 = pred_normal_v2
        
    ax4.scatter(x5, y5, z5, color='r', marker='o')

    # get theta from 2 vectors
    theta_radian = math.acos(np.dot(gt_normal_v1, pred_normal_v1) / (np.linalg.norm(gt_normal_v1) * np.linalg.norm(pred_normal_v1)))
    theta_degree = math.degrees(theta_radian)
    theta_list = [theta_degree, 0, 0, 0]
    

    # We need to compare 2 gt_normal_vector & 2 pred_normal_vector. Orientation error cannot exceed 45 degree.
    if theta_degree > 45:
        theta_radian = math.acos(np.dot(gt_normal_v1, pred_normal_v2) / (np.linalg.norm(gt_normal_v1) * np.linalg.norm(pred_normal_v2)))
        theta_degree = math.degrees(theta_radian)
        theta_list[1] = theta_degree

        x5, x6 = x6, x5
        y5, y6 = y6, y5
        z5, z6 = z6, z5
        x7, x8 = x8, x7
        y7, y8 = y8, y7
        z7, z8 = z8, z7

        if theta_degree > 45:
            theta_radian = math.acos(np.dot(gt_normal_v2, pred_normal_v2) / (np.linalg.norm(gt_normal_v2) * np.linalg.norm(pred_normal_v2)))
            theta_degree = math.degrees(theta_radian)
            theta_list[2] = theta_degree
            
            u1, u2 = u2, u1
            v1, v2 = v2, v1
            w1, w2 = w2, w1
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            z1, z2 = z2, z1

            if theta_degree > 45:
                
                theta_radian = math.acos(np.dot(gt_normal_v2, pred_normal_v1) / (np.linalg.norm(gt_normal_v2) * np.linalg.norm(pred_normal_v1)))
                theta_degree = math.degrees(theta_radian)
                theta_list[3] = theta_degree

                x5, x6 = x6, x5
                y5, y6 = y6, y5
                z5, z6 = z6, z5
                x7, x8 = x8, x7
                y7, y8 = y8, y7
                z7, z8 = z8, z7

                #if theta_degree > 45:
                    #all orientation difference between two planes considered front > 45
                    # Then define orientation difference as the smallest one.

                #    theta_degree = min(theta_list)
                    #theta_index = theta_degree.index(theta_degree)





    ax4.quiver(u1, v1, w1, x1, y1, z1, color="b", arrow_length_ratio=0.2, length=0.3, normalize=True)
    ax4.quiver(x5, y5, z5, x7, y7, z7, color="r", arrow_length_ratio=0.2, length=0.3, normalize=True)
    #ax4.quiver(x6, y6, z6, x8, y8, z8, color="r", arrow_length_ratio=0.2, length=0.3, normalize=True)

    orientation_error = theta_degree
    orientation_errors.append(orientation_error)

print("Orientation Errors: \n{}\n{}\n".format(np.mean(orientation_errors), np.max(orientation_errors)))

# fig3 = plt.figure()
# ax5 = fig3.add_subplot(1,1,1)
# ax5.hist(add,bins=10,label="ADD_4 error")
# ax5.set_title("Azure kinect DK ADD4 error \n Random Boxes: distance = 1900mm  n = {}".format(len(add)))
# ax5.set_xlabel("ADD error")
# ax5.set_ylabel("Sample count")

ax1.set_title("Predicted Positions")
ax2.set_title("Ground Truth Positions")
# ax3.set_title("True Associations")
ax4.set_title("Founds Associations")


ax1.set_box_aspect([-1,-1,1])
ax2.set_box_aspect([-1,-1,1])
# ax3.set_box_aspect([-1,-1,1])
ax4.set_box_aspect([-1,-1,1])

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

ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("z")

set_axes_equal(ax1)
set_axes_equal(ax2)
# set_axes_equal(ax3)
set_axes_equal(ax4)

plt.tight_layout()
plt.show()
# plt.savefig('/home/ksy/azure_pouch_box_1670mm.png')
# plt.savefig('/home/ksy/azure_sack_box_pouch_1500mm.png')
# plt.savefig('/home/ksy/azure_scramble_1900mm.png')
# plt.savefig('/home/ksy/azure_stacked_boxes_2400mm.png')
# plt.savefig('/home/ksy/ensenso_sack_box_pouch_3600mm.png')
# plt.savefig('/home/ksy/ensenso_scramble_2400mm.png')
# plt.savefig('/home/ksy/ensenso_stacked_boxes_2700mm.png')
# plt.savefig('/home/ksy/mechmind_free_form_2400mm.png')
# plt.savefig('/home/ksy/mechmind_stacked_boxes_3300mm.png')
