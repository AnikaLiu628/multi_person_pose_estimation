from time import time

import cv2
import numpy as np
import scipy
import tensorflow as tf


class Preprocess():
    def __init__(self, kernel_size=4):
        self.skeleton = [
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
            (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
            (2, 4), (3, 5), (4, 6), (5, 7),
        ]
        self.kernel_size = kernel_size
        self.sink = self.create_sink(self.kernel_size)

    def pyfn_interface_input(self, parsed_features):
        return parsed_features['image/decoded'], parsed_features['image/filename'], \
            parsed_features['image/height'], parsed_features['image/width'], \
            parsed_features['image/human/bbox/xmin'], parsed_features['image/human/bbox/xmax'], \
            parsed_features['image/human/bbox/ymin'], parsed_features['image/human/bbox/ymax'], \
            parsed_features['image/human/keypoints/x'], parsed_features['image/human/keypoints/y'], \
            parsed_features['image/human/keypoints/v'], parsed_features['image/human/num_keypoints']

    def pyfn_interface_output(self, img, source,
                              pif_intensities, pif_fields_reg, pif_fields_scale,
                              paf_intensities, paf_fields_reg1, paf_fields_reg2,
                              keypoint_sets):
        parsed_features = {
            'image/decoded': img, 'image/filename': source,
            'image/pif/intensities': pif_intensities, 'image/pif/fields_reg': pif_fields_reg,
            'image/pif/fields_scale': pif_fields_scale, 'image/paf/intensities': paf_intensities,
            'image/paf/fields_reg1': paf_fields_reg1, 'image/paf/fields_reg2': paf_fields_reg2,
            'image/keypoint_sets': keypoint_sets,
        }
        return parsed_features

    def create_sink(self, side):
            if side == 1:
                return np.zeros((2, 1, 1))

            sink1d = np.linspace((side - 1.0) / 2.0, -(side - 1.0) / 2.0, num=side, dtype=np.float32)
            sink = np.stack((
                sink1d.reshape(1, -1).repeat(side, axis=0),
                sink1d.reshape(-1, 1).repeat(side, axis=1),
            ), axis=0)
            return sink

    def head_encoder(self, img, source, height, width,
                     bbx1, bbx2, bby1, bby2,
                     kx, ky, kv, nkp):
        w = 640
        h = 360
        scaling_ratio = 8
        width_ratio = width / w
        height_ratio = height / h
        side_length = 4.0
        s_offset = (side_length - 1.0) / 2.0
        padding = 10
        field_h = int(h / scaling_ratio + 2 * padding)
        field_w = int(w / scaling_ratio + 2 * padding)
        num_keypoints = 17
        bx1 = np.reshape(bbx1, [-1, 1])
        bx1 = bx1.astype(np.float64) / (width_ratio * scaling_ratio)
        bx2 = np.reshape(bbx2, [-1, 1])
        bx2 = bx2.astype(np.float64) / (width_ratio * scaling_ratio)
        by1 = np.reshape(bby1, [-1, 1])
        by1 = by1.astype(np.float64) / (height_ratio * scaling_ratio)
        by2 = np.reshape(bby2, [-1, 1])
        by2 = by2.astype(np.float64) / (height_ratio * scaling_ratio)
        bbox = np.concatenate([bx1, bx2, by1, by2], axis=1).astype(np.int32) + padding

        xs = np.reshape(kx, [-1, 1])
        ys = np.reshape(ky, [-1, 1])
        vs = np.reshape(kv, [-1, 1])
        xs = xs.astype(np.float64) / (width_ratio * scaling_ratio)
        ys = ys.astype(np.float64) / (height_ratio * scaling_ratio)
        keypoint_sets = np.concatenate([xs, ys], axis=1)
        keypoint_sets = np.concatenate([keypoint_sets, vs.astype(np.float64)], axis=1)
        keypoint_sets = keypoint_sets.astype(np.float32)

        bg = np.ones((1, field_h, field_w), dtype=np.float32)
        for xxyy in bbox:
            bg[:, xxyy[2]:xxyy[3], xxyy[0]:xxyy[1]] = 0

        bg = scipy.ndimage.binary_erosion(bg, iterations=2, border_value=1)

        pif_intensities = np.zeros((num_keypoints, field_h, field_w), dtype=np.float32)
        pif_fields_reg = np.zeros((num_keypoints, 2, field_h, field_w), dtype=np.float32)
        pif_fields_reg_l = np.full((num_keypoints, field_h, field_w), np.inf, dtype=np.float32)
        pif_fields_scale = np.zeros((num_keypoints, field_h, field_w), dtype=np.float32)  
        keypoint_sets = np.reshape(keypoint_sets, [-1, num_keypoints, 3])
        self.pif_generator(pif_intensities, pif_fields_reg, pif_fields_reg_l, pif_fields_scale, keypoint_sets, vs)

        paf_n_fields = 19
        paf_intensities = np.zeros((paf_n_fields, field_h, field_w), dtype=np.float32)
        paf_fields_reg1 = np.zeros((paf_n_fields, 2, field_h, field_w), dtype=np.float32)
        paf_fields_reg2 = np.zeros((paf_n_fields, 2, field_h, field_w), dtype=np.float32)
        paf_fields_reg_l = np.full((paf_n_fields, field_h, field_w), np.inf, dtype=np.float32)
        self.paf_generator(paf_intensities, paf_fields_reg1, paf_fields_reg2, paf_fields_reg_l, keypoint_sets)

        pif_intensities = np.concatenate([pif_intensities, bg], axis=0)
        paf_intensities = np.concatenate([paf_intensities, bg], axis=0)

        return img, source, \
           pif_intensities[:, padding:-padding, padding:-padding], \
           pif_fields_reg[:, :, padding:-padding, padding:-padding], \
           pif_fields_scale[:, padding:-padding, padding:-padding], \
           paf_intensities[:, padding:-padding, padding:-padding], \
           paf_fields_reg1[:, :, padding:-padding, padding:-padding], \
           paf_fields_reg2[:, :, padding:-padding, padding:-padding], \
           keypoint_sets
           
    def pif_generator(self, intensities, fields_reg, fields_reg_l, fields_scale, keypoint_sets, vs, num_keypoints=17, padding=10):
        for keypoints in keypoint_sets:
            bool_mask = keypoints[:, 2] > 0
            if not np.any(bool_mask):
                continue

            omit_zeros = keypoints[bool_mask]
            valid_p_minx = np.amin(omit_zeros[:, 0])
            valid_p_maxx = np.amax(omit_zeros[:, 0])
            valid_p_miny = np.amin(omit_zeros[:, 1])
            valid_p_maxy = np.amax(omit_zeros[:, 1])
            scale = (valid_p_maxx - valid_p_minx) * (valid_p_maxy - valid_p_miny)
            scale = np.sqrt(scale)
            for f, xyv in enumerate(keypoints):
                if xyv[2] <= 0:
                    continue

                self.fill_coordinate(f, xyv, scale, intensities, fields_reg, fields_reg_l, fields_scale)

    def fill_coordinate(self, f, xyv, scale, intensities, fields_reg, fields_reg_l, fields_scale, padding=10):
        ij = np.round(xyv[:2] - 1.5).astype(np.int) + padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + 4, miny + 4
        if minx < 0 or maxx > intensities.shape[2] or \
           miny < 0 or maxy > intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + 1.5 - 10)
        offset = offset.reshape(2, 1, 1)

        # update intensity
        intensities[f, miny:maxy, minx:maxx] = 1.0

        # update regression
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < fields_reg_l[f, miny:maxy, minx:maxx]
        fields_reg[f, :, miny:maxy, minx:maxx][:, mask] = \
            sink_reg[:, mask]
        fields_reg_l[f, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update scale
        fields_scale[f, miny:maxy, minx:maxx][mask] = scale

    def paf_generator(self, intensities, fields_reg1, fields_reg2, fields_reg_l, keypoint_sets):
        for keypoints in keypoint_sets:
            for i, (joint1i, joint2i) in enumerate(self.skeleton):
                joint1 = keypoints[joint1i - 1]
                joint2 = keypoints[joint2i - 1]
                if joint1[2] <= 0 or joint2[2] <= 0:
                    continue

                self.fill_association(i, joint1, joint2, intensities, fields_reg1, fields_reg2, fields_reg_l)

    def fill_association(self, i, joint1, joint2, intensities, fields_reg1, fields_reg2, fields_reg_l, padding=10):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = 3
        sink = self.create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # pixel coordinates of top-left joint pixel
        joint1ij = np.round(joint1[:2] - s_offset)
        joint2ij = np.round(joint2[:2] - s_offset)
        offsetij = joint2ij - joint1ij

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0-fmargin, num=num)

        for f in frange:
            fij = np.round(joint1ij + f * offsetij) + padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > intensities.shape[2] or \
               fminy < 0 or fmaxy > intensities.shape[1]:
                continue
            fxy = (fij - padding) + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)

            # update intensity
            intensities[i, fminy:fmaxy, fminx:fmaxx] = 1.0

            # update regressions
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset
            sink_l = np.minimum(np.linalg.norm(sink1, axis=0),
                                np.linalg.norm(sink2, axis=0))
            mask = sink_l < fields_reg_l[i, fminy:fmaxy, fminx:fmaxx]
            fields_reg1[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink1[:, mask]
            fields_reg2[i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink2[:, mask]
            fields_reg_l[i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

