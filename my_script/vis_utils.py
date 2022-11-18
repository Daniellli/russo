

#!/usr/bin/env python
#!coding=utf-8
import numpy as np
import open3d


'''
description:  open3d data to numpy  
param undefined
return {*}
'''
def open3d2numpy(data):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(data)
    return np.asarray(pcd.points)



'''
description:  numpy data to open3d  
param undefined
return {*}
'''
def numpy2open3d(np_points):

    pcd = open3d.geometry.PointCloud()
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(np_points)

    return pcd


'''
description: 
param undefined
return {*}
'''
def numpy2open3d_colorful(np_points):

    pcd = open3d.geometry.PointCloud()
    # From numpy to Open3D
    pcd.points = open3d.utility.Vector3dVector(np_points[:,:3])
    pcd.colors = open3d.utility.Vector3dVector(np_points[:,3:])

    return pcd
        





def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
    
    
    

'''
description:  given center point and box size (l,h,w) and normal angle generating the bounding box 8 corner  coordination  
param {*} center
param {*} size
param {*} heading_angle
return {*}
'''
def my_compute_box_3d(center, size, heading_angle):

    h = size[2]
    w = size[0]
    l = size[1]
    heading_angle = -heading_angle - np.pi / 2 

    center[2] = center[2] + h / 2
    R = rotz(1*heading_angle)
    
    l = l/2
    w = w/2
    h = h/2
    x_corners = [-l,l,l,-l,-l,l,l,-l]
    y_corners = [w,w,-w,-w,w,w,-w,-w]
    z_corners = [h,h,h,h,-h,-h,-h,-h]
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0,:] += center[0]
    corners_3d[1,:] += center[1]
    corners_3d[2,:] += center[2]
    return np.transpose(corners_3d)





'''
description: draw point cloud and pred box   and gt box 
param1 pc : point cloud  
param1 bbox :  bounding box 
param1 gt_bbox :  ground truth bounding box 
param1 pred_box_color :   predict box color 
param1 gt_box_color :   ground truth bounding box color 
save_path: save file name path and name 
return {*}
'''
def draw_pc_box(pc,bbox,gt_bbox = None,
                pred_box_color=[1, 0, 0], 
                gt_box_color=[0,1,0],
                save_path="open3d.png"):

    # axis = open3d.create_mesh_coordinate_frame(size=1,origin=[0,0,0])

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 10

    # ctr = vis.get_view_control()
    # ctr.set_lookat([0, 0, 0.3])
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])

    
    if gt_bbox is not None:
        for i in range(len(gt_bbox)):
            bbox = gt_bbox[i]
            
            corners_3d = my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])

            bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                          [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]
            
            colors = [  gt_box_color for _ in range(len(bbox_lines))] #red
            
            bbox = open3d.geometry.LineSet()
            bbox.lines  = open3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = open3d.utility.Vector3dVector(colors)
            bbox.points = open3d.utility.Vector3dVector(corners_3d)
            vis.add_geometry(bbox)

    #* add 3D  bounding box 
    for i in range(len(boxes)):
        bbox = boxes[i]
        
        corners_3d = my_compute_box_3d(bbox[0:3], bbox[3:6],0)


        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                      [4, 5], [5, 6], [6, 7], [7, 4], 
                      [0, 4], [1, 5], [2, 6], [3, 7]]
        
        colors = [pred_box_color for _ in range(len(bbox_lines))]  #green

        bbox = open3d.geometry.LineSet()
        bbox.lines  = open3d.utility.Vector2iVector(bbox_lines)
        bbox.colors = open3d.utility.Vector3dVector(colors)
        bbox.points = open3d.utility.Vector3dVector(corners_3d)
        vis.add_geometry(bbox)

    vis.add_geometry(pc)
    
    
    controler = vis.get_view_control()
    controler.rotate(200,-250)
    # vis.run()
    # vis.update_geometry(source)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    # vis.destroy_window()
    
        
        
        
if __name__ == '__main__':

    path = "/Users/xushaocong/Downloads/butd_detr/ddd/scene0017_02_pc.txt"
    data = np.loadtxt(path,dtype=np.float64)
    pointcloud = numpy2open3d_colorful(data)
    
    boxes_p = "/Users/xushaocong/Downloads/butd_detr/ddd/scene0017_02_box.txt"
    boxes = np.loadtxt(boxes_p,dtype=np.float64)
    draw_pc_box(pointcloud,boxes)

