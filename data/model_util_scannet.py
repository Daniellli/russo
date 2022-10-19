'''
Author: xushaocong
Date: 2022-08-19 18:59:01
LastEditTime: 2022-10-18 17:16:06
LastEditors: xushaocong
Description: 
FilePath: /butd_detr/data/model_util_scannet.py
email: xushaocong@stu.xmu.edu.cn
'''
# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

import numpy as np


class ScannetDatasetConfig:

    def __init__(self, num_class=485, agnostic=False):
        self.num_class = num_class if not agnostic else 1  # 18
        self.num_heading_bin = 1
        self.num_size_cluster = num_class
        if num_class == 18:
            self.type2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'couch': 3, 'table': 4, 'door': 5,
                               'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                               'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16,
                               'other furniture': 17}
        else:
            self.type2class = {'wall': 0, 'chair': 1, 'floor': 2, 'table': 3, 'door': 4, 'couch': 5, 'cabinet': 6, 
            'shelf': 7, 'desk': 8, 'office chair': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 
            'picture': 13, 'window': 14, 'toilet': 15, 'bookshelf': 16, 'monitor': 17, 
            'curtain': 18, 'book': 19, 'armchair': 20, 'coffee table': 21, 'drawer': 22,
            'box': 23, 'refrigerator': 24, 'lamp': 25, 'kitchen cabinet': 26, 'towel': 27,
            'clothes': 28, 'tv': 29, 'nightstand': 30, 'counter': 31, 'dresser': 32, 
            'stool': 33, 'couch cushions': 34, 'plant': 35, 'ceiling': 36, 'bathtub': 37, 
            'end table': 38, 'dining table': 39, 'keyboard': 40, 'bag': 41, 'backpack': 42,
            'toilet paper': 43, 'printer': 44, 'tv stand': 45, 'whiteboard': 46, 'carpet': 47,
            'blanket': 48, 'shower curtain': 49, 'trash can': 50, 'closet': 51, 'staircase': 52, 
            'microwave': 53, 'rug': 54, 'stove': 55, 'shoe': 56, 'computer tower': 57, 'bottle': 58,
            'bin': 59, 'ottoman': 60, 'bench': 61, 'board': 62, 'washing machine': 63, 'mirror': 64, 
            'copier': 65, 'basket': 66, 'sofa chair': 67, 'file cabinet': 68, 'fan': 69, 'laptop': 70,
            'shower': 71, 'paper': 72, 'person': 73, 'headboard': 74, 'paper towel dispenser': 75, 
            'faucet': 76, 'oven': 77, 'footstool': 78, 'blinds': 79, 'rack': 80, 'plate': 81, 'blackboard': 82,
            'piano': 83, 'heater': 84, 'soap': 85, 'suitcase': 86, 'rail': 87, 'radiator': 88, 
            'recycling bin': 89, 'container': 90, 'closet wardrobe': 91, 'soap dispenser': 92,
            'telephone': 93, 'bucket': 94, 'clock': 95, 'stand': 96, 'light': 97, 'laundry basket': 98,
            'pipe': 99, 'round table': 100, 'clothes dryer': 101, 'coat': 102, 'guitar': 103,
            'toilet paper holder': 104, 'seat': 105, 'step': 106, 'speaker': 107, 'vending machine': 108,
            'column': 109, 'bicycle': 110, 'ladder': 111, 'cover': 112, 'bathroom stall': 113, 
            'foosball table': 114, 'shower wall': 115, 'chest': 116, 'cup': 117, 'jacket': 118,
            'storage bin': 119, 'screen': 120, 'coffee maker': 121, 'hamper': 122, 'dishwasher': 123,
            'paper towel roll': 124, 'machine': 125, 'mat': 126, 'windowsill': 127, 'tap': 128, 
            'pool table': 129, 'hand dryer': 130, 'bar': 131, 'frame': 132, 'toaster': 133,
            'handrail': 134, 'bulletin board': 135, 'ironing board': 136, 'fireplace': 137, 
            'soap dish': 138, 'kitchen counter': 139, 'glass': 140, 'doorframe': 141, 
            'toilet paper dispenser': 142, 'mini fridge': 143, 'fire extinguisher': 144,
            'shampoo bottle': 145, 'ball': 146, 'hat': 147, 'shower curtain rod': 148,
            'toiletry': 149, 'water cooler': 150, 'desk lamp': 151, 'paper cutter': 152, 
            'switch': 153, 'tray': 154, 'shower door': 155, 'shirt': 156, 'pillar': 157,
            'ledge': 158, 'vase': 159, 'toaster oven': 160, 'mouse': 161, 'nerf gun': 162, 
            'toilet seat cover dispenser': 163, 'can': 164, 'furniture': 165, 'cart': 166, 
            'step stool': 167, 'dispenser': 168, 'storage container': 169, 'side table': 170,
            'lotion': 171, 'cooking pot': 172, 'toilet brush': 173, 'scale': 174, 
            'tissue box': 175, 'remote': 176, 'light switch': 177, 'crate': 178, 
            'ping pong table': 179, 'platform': 180, 'slipper': 181, 'power outlet': 182, 
            'cutting board': 183, 'controller': 184, 'decoration': 185, 'trolley': 186,
            'sign': 187, 'projector': 188, 'sweater': 189, 'globe': 190, 'closet door': 191,
            'plastic container': 192, 'statue': 193, 'vacuum cleaner': 194, 'wet floor sign': 195,
            'candle': 196, 'easel': 197, 'wall hanging': 198, 'dumbell': 199, 'ping pong paddle': 200,
            'plunger': 201, 'soap bar': 202, 'stuffed animal': 203, 'water fountain': 204, 
            'footrest': 205, 'headphones': 206, 'plastic bin': 207, 'coatrack': 208,
            'dish rack': 209, 'broom': 210, 'guitar case': 211, 'mop': 212, 'magazine': 213,
            'range hood': 214, 'scanner': 215, 'bathrobe': 216, 'futon': 217, 'dustpan': 218,
            'hand towel': 219, 'organizer': 220, 'map': 221, 'helmet': 222, 'hair dryer': 223,
            'exercise ball': 224, 'iron': 225, 'studio light': 226, 'cabinet door': 227,
            'exercise machine': 228, 'workbench': 229, 'water bottle': 230, 'handicap bar': 231, 'tank': 232, 
            'purse': 233, 'vent': 234, 'piano bench': 235, 'bunk bed': 236, 'shoe rack': 237, 'shower floor': 238, 
            'case': 239, 'swiffer': 240, 'stapler': 241, 'cable': 242, 'garbage bag': 243, 'banister': 244, 
            'trunk': 245, 'tire': 246, 'folder': 247, 'car': 248, 'flower stand': 249, 'water pitcher': 250,
            'loft bed': 251, 'shopping bag': 252, 'curtain rod': 253, 'alarm': 254, 'washcloth': 255, 
            'toolbox': 256, 'sewing machine': 257, 'mailbox': 258, 'toothpaste': 259, 'rope': 260,
            'electric panel': 261, 'bowl': 262, 'boiler': 263, 'paper bag': 264, 'alarm clock': 265,
            'music stand': 266, 'instrument case': 267, 'paper tray': 268, 'paper shredder': 269,
            'projector screen': 270, 'boots': 271, 'kettle': 272, 'mail tray': 273, 'cat litter box': 274, 
            'covered box': 275, 'ceiling fan': 276, 'cardboard': 277, 'binder': 278, 'beachball': 279,
            'envelope': 280, 'thermos': 281, 'breakfast bar': 282, 'dress rack': 283, 'frying pan': 284,
            'divider': 285, 'rod': 286, 'magazine rack': 287, 'laundry detergent': 288, 'sofa bed': 289,
            'storage shelf': 290, 'loofa': 291, 'bycicle': 292, 'file organizer': 293, 'fire hose': 294,
            'media center': 295, 'umbrella': 296, 'barrier': 297, 'subwoofer': 298, 'stepladder': 299, 
            'shorts': 300, 'rocking chair': 301, 'elliptical machine': 302, 'coffee mug': 303, 'jar': 304,
            'door wall': 305, 'traffic cone': 306, 'pants': 307, 'garage door': 308, 'teapot': 309, 
            'barricade': 310, 'exit sign': 311, 'canopy': 312, 'kinect': 313, 'kitchen island': 314,
            'messenger bag': 315, 'buddha': 316, 'block': 317, 'stepstool': 318, 'tripod': 319, 
            'chandelier': 320, 'smoke detector': 321, 'baseball cap': 322, 'toothbrush': 323,
            'bathroom counter': 324, 'object': 325, 'bathroom vanity': 326, 'closet wall': 327, 
            'laundry hamper': 328, 'bathroom stall door': 329, 'ceiling light': 330, 'trash bin': 331,
            'dumbbell': 332, 'stair rail': 333, 'tube': 334, 'bathroom cabinet': 335, 'cd case': 336,
            'closet rod': 337, 'coffee kettle': 338, 'wardrobe cabinet': 339, 'structure': 340, 
            'shower head': 341, 'keyboard piano': 342, 'case of water bottles': 343, 'coat rack': 344,
            'storage organizer': 345, 'folded chair': 346, 'fire alarm': 347, 'power strip': 348, 
            'calendar': 349, 'poster': 350, 'potted plant': 351, 'luggage': 352, 'mattress': 353,
            'hand rail': 354, 'folded table': 355, 'poster tube': 356, 'thermostat': 357,
            'flip flops': 358, 'cloth': 359, 'banner': 360, 'clothes hanger': 361,
            'whiteboard eraser': 362, 'shower control valve': 363, 'compost bin': 364, 
            'teddy bear': 365, 'pantry wall': 366, 'tupperware': 367, 'beer bottles': 368,
            'salt': 369, 'mirror doors': 370, 'folded ladder': 371, 'carton': 372,
            'soda stream': 373, 'metronome': 374, 'music book': 375, 'rice cooker': 376,
            'dart board': 377, 'grab bar': 378, 'flowerpot': 379, 'painting': 380, 'railing': 381, 
            'stair': 382, 'quadcopter': 383, 'pitcher': 384, 'hanging': 385, 'mail': 386, 'closet ceiling': 387, 
            'hoverboard': 388, 'beanbag chair': 389, 'spray bottle': 390, 'soap bottle': 391, 'ikea bag': 392,
            'duffel bag': 393, 'oven mitt': 394, 'pot': 395, 'hair brush': 396, 'tennis racket': 397, 'display case': 398, 
            'bananas': 399, 'carseat': 400, 'coffee box': 401, 'clothing rack': 402, 'bath walls': 403, 'podium': 404, 
            'storage box': 405, 'dolly': 406, 'shampoo': 407, 'changing station': 408, 'crutches': 409, 'grocery bag': 410,
            'pizza box': 411, 'shaving cream': 412, 'luggage rack': 413, 'urinal': 414, 'hose': 415, 'bike pump': 416,
            'bear': 417, 'humidifier': 418, 'mouthwash bottle': 419, 'golf bag': 420, 'food container': 421, 'card': 422,
            'mug': 423, 'boxes of paper': 424, 'flag': 425, 'rolled poster': 426, 'wheel': 427, 'blackboard eraser': 428, 
            'doll': 429, 'laundry bag': 430, 'sponge': 431, 'lotion bottle': 432, 'lunch box': 433, 'sliding wood door': 434,
            'briefcase': 435, 'bath products': 436, 'star': 437, 'coffee bean bag': 438, 'ipad': 439, 'display rack': 440,
            'massage chair': 441, 'paper organizer': 442, 'cap': 443, 'dumbbell plates': 444, 'elevator': 445, 'cooking pan': 446, 
            'trash bag': 447, 'santa': 448, 'jewelry box': 449, 'boat': 450, 'sock': 451, 'plastic storage bin': 452, 'dishwashing soap bottle': 453, 
            'xbox controller': 454, 'airplane': 455, 'conditioner bottle': 456, 'tea kettle': 457, 'wall mounted coat rack': 458, 
            'film light': 459, 'sofa': 460, 'pantry shelf': 461, 'fish': 462, 'toy dinosaur': 463, 'cone': 464, 
            'fire sprinkler': 465, 'contact lens solution bottle': 466, 'hand sanitzer dispenser': 467,
            'pen holder': 468, 'wig': 469, 'night light': 470, 'notepad': 471, 'drum set': 472,
            'closet shelf': 473, 'exercise bike': 474, 'soda can': 475, 'stovetop': 476, 'telescope': 477, 
            'battery disposal jar': 478, 'closet floor': 479, 'clip': 480, 'display': 481, 'postcard': 482, 
            'paper towel': 483, 'food bag': 484}


        self.class2type = {self.type2class[t]: t for t in self.type2class}

        if num_class == 18:
            self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])
        else:
            # only train
            self.nyu40ids = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125, 126, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145, 146, 148, 152, 154, 155, 156, 157, 159, 160, 161, 163, 165, 166, 167, 168, 169, 170, 174, 177, 179, 180, 182, 185, 188, 189, 191, 193, 194, 195, 202, 204, 208, 212, 213, 214, 216, 220, 221, 222, 225, 226, 228, 229, 230, 231, 232, 233, 234, 235, 238, 242, 245, 247, 250, 257, 261, 264, 265, 269, 276, 280, 281, 283, 284, 286, 289, 291, 297, 298, 300, 301, 304, 305, 307, 312, 316, 319, 323, 325, 331, 332, 339, 342, 345, 346, 354, 356, 357, 361, 365, 366, 370, 372, 378, 379, 385, 386, 389, 392, 395, 397, 399, 408, 410, 411, 415, 417, 432, 434, 435, 436, 440, 448, 450, 452, 459, 461, 484, 488, 494, 506, 513, 518, 523, 525, 529, 540, 546, 556, 561, 562, 563, 570, 572, 581, 591, 592, 599, 609, 612, 621, 643, 657, 673, 682, 689, 693, 712, 719, 726, 730, 733, 746, 748, 750, 765, 776, 786, 794, 801, 803, 813, 814, 815, 816, 817, 819, 851, 857, 885, 893, 907, 919, 947, 948, 955, 976, 997, 1005, 1009, 1028, 1051, 1063, 1072, 1083, 1098, 1116, 1117, 1122, 1125, 1126, 1135, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172, 1173, 1174, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1198, 1199, 1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207, 1208, 1209, 1210, 1211, 1212, 1213, 1214, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1225, 1226, 1227, 1228, 1229, 1230, 1232, 1233, 1234, 1235, 1236, 1237, 1239, 1240, 1241, 1242, 1243, 1244, 1245, 1246, 1247, 1248, 1250, 1252, 1253, 1255, 1256, 1257, 1258, 1259, 1260, 1261, 1262, 1264, 1265, 1268, 1269, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1282, 1285, 1286, 1287, 1288, 1289, 1290, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1304, 1305, 1307, 1308, 1309, 1311, 1312, 1313, 1316, 1318, 1319, 1320, 1321, 1324, 1326, 1327, 1329, 1330, 1331, 1334, 1335, 1337, 1339, 1340, 1344, 1346, 1347, 1350, 1351, 1352, 1353, 1356])
            
        self.nyu40id2class = {nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))}


#!+============================================================================

def rotate_aligned_boxes(input_boxes, rot_mat):    
    centers, lengths = input_boxes[:,0:3], input_boxes[:,3:6]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
           
    dx, dy = lengths[:,0]/2.0, lengths[:,1]/2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    
    for i, crnr in enumerate([(-1,-1), (1, -1), (1, 1), (-1, 1)]):        
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:,0] = crnr[0]*dx
        crnrs[:,1] = crnr[1]*dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:,i] = crnrs[:,0]
        new_y[:,i] = crnrs[:,1]
    
    
    new_dx = 2.0*np.max(new_x, 1)
    new_dy = 2.0*np.max(new_y, 1)    
    new_lengths = np.stack((new_dx, new_dy, lengths[:,2]), axis=1)
                  
    return np.concatenate([new_centers, new_lengths], axis=1)


def rotate_quad(rectangle, rot_mat):    
    centers = rectangle[:,0:3]    
    new_centers = np.dot(centers, np.transpose(rot_mat))
    normal_vector = rectangle[:,3:6]
    new_normal_vector = np.dot(normal_vector, np.transpose(rot_mat))
                  
    return np.concatenate([new_centers, new_normal_vector,rectangle[:,6:8]], axis=1)


#!+============================================================================