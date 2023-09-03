from __future__ import print_function, absolute_import, division
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this SHIFT label.

    'cityscapesId', # Equivalent class ID in Cityscapes labels

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ]
)

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------
labels = [
	Label("unlabeled"     ,  0 ,      0,	    True,  (  0,   0,   0)),
	Label("building"      ,  1 ,     11,		False, ( 70,  70,  70)),
	Label("fence"         ,  2 ,     13,		False, (100,  40,  40)),
	Label("other"         ,  3 ,      0,		True,  ( 55,  90,  80)),
	Label("pedestrian"    ,  4 ,     24,		False, (220,  20,  60)),
	Label("pole"          ,  5 ,     17,		False, (153, 153, 153)),
	Label("road line"     ,  6 ,      7,		False, (157, 234,  50)),
	Label("road"          ,  7 ,      7,		False, (128,  64, 128)),
	Label("sidewalk"      ,  8 ,      8,		False, (244,  35, 232)),
	Label("vegetation"    ,  9 ,     21,		False, (107, 142,  35)),
	Label("vehicle"       , 10 ,     26,		False, (  0,   0, 142)),
	Label("wall"          , 11 ,     12,		False, (102, 102, 156)),
	Label("traffic sign"  , 12 ,     20,		False, (220, 220,   0)),
	Label("sky"           , 13 ,     23,		False, ( 70, 130, 180)),
	Label("ground"        , 14 ,      6,		True,  ( 81,   0,  81)),
	Label("bridge"        , 15 ,     15,		True,  (150, 100, 100)),
	Label("rail track"    , 16 ,     10,		True,  (230, 150, 140)),
	Label("guard rail"    , 17 ,     14,		True,  (180, 165, 180)),
	Label("traffic light" , 18 ,     19,		False, (250, 170,  30)),
	Label("static"        , 19 ,      4,		True,  (110, 190, 160)),
	Label("dynamic"       , 20 ,      5,		True,  (170, 120,  50)),
	Label("water"         , 21 ,      0,		True,  ( 45,  60, 150)),
	Label("terrain"       , 22 ,     22,		False, (145, 170, 100)),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
shift2cityscapes  = { label.id : label.cityscapesId for label in labels }
