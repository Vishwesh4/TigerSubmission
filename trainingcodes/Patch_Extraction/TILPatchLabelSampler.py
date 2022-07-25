from typing import List, Optional

import cv2
import numpy as np
from wholeslidedata.annotation.structures import Point, Polygon
from wholeslidedata.samplers.utils import shift_coordinates
from wholeslidedata.samplers.patchlabelsampler import PatchLabelSampler

@PatchLabelSampler.register(("segmentation",))
class TILPatchLabelSampler(PatchLabelSampler):
    def __init__(self):
        pass

    # annotation should be coupled to image_annotation. how?
    def sample(
        self,
        wsa,
        point,
        size,
        ratio,
    ):
        center_x, center_y = point.x, point.y
        width, height = size

        # get annotations
        annotations = wsa.select_annotations(
            center_x, center_y, (width * ratio) - 1, (height * ratio) - 1
        )

        # create mask placeholder
        mask = np.zeros((height, width), dtype=np.int32)
        mask_cell = np.zeros((height, width), dtype=np.int32)
        # set labels of all selected annotations
        for annotation in annotations:
            coordinates = np.copy(annotation.coordinates)
            coordinates = shift_coordinates(
                coordinates, center_x, center_y, width, height, ratio
            )

            if isinstance(annotation, Polygon):
                if annotation.label._name=='lymphocytes and plasma cells':
                    #Convert box into point annotation
                #    box_x = int(np.mean(coordinates[:4,0]))
                #    box_y = int(np.mean(coordinates[:4,1]))
                #    mask[box_x,box_y] = annotation.label.value
                    holemask = np.ones((height, width), dtype=np.int32) * -1
                    cv2.fillPoly(
                        mask_cell,
                        np.array([coordinates], dtype=np.int32),
                        1,
                    )
                    mask_cell[holemask != -1] = holemask[holemask != -1]
                else:
                    holemask = np.ones((height, width), dtype=np.int32) * -1
                    for hole in annotation.holes:
                        hcoordinates = shift_coordinates(
                            hole, center_x, center_y, width, height, ratio
                        )
                        cv2.fillPoly(holemask, np.array([hcoordinates], dtype=np.int32), 1)
                        holemask[holemask != -1] = mask[holemask != -1]
                    cv2.fillPoly(
                        mask,
                        np.array([coordinates], dtype=np.int32),
                        annotation.label.value,
                    )
                    mask[holemask != -1] = holemask[holemask != -1]

            elif isinstance(annotation, Point):
                mask[int(coordinates[1]), int(coordinates[0])] = annotation.label.value

        return [mask.astype(np.uint8),mask_cell.astype(np.uint8)]

