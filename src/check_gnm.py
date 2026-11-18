import numpy as np
from gnm.shape import gnm_numpy

gnm = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)

identity = np.zeros(gnm.identity_dim)            # zeros = the average (neutral) head
expression = np.zeros(gnm.expression_dim)
rotations = np.zeros((gnm.num_joints, 3))
translation = np.zeros(3)
vertices = gnm(identity, expression, rotations, translation)

print("identity parameters:", gnm.identity_dim, "| expression parameters:", gnm.expression_dim)
print("vertices:", vertices.shape, "| triangles:", gnm.triangles.shape)
print("head size (mm or model units?):", np.ptp(vertices, axis=0).round(3))