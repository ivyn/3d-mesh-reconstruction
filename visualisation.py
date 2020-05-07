# Written by Pransu Dash, Ryan Leung, and Ivy Nguyen

import numpy as np
import open3d as o3d
from bpa import BPA
from bpa import Triangle


if __name__ == "__main__":

	print("Load a ply point cloud, print it, and render it")
	pcd = o3d.io.read_point_cloud("madara.ply")
	pcd.estimate_normals(fast_normal_computation=False)

	pcd.normalize_normals()
	pcd.orient_normals_to_align_with_direction()
	b = BPA(pcd)
	print('making mesh')
	triangles = b.make_mesh()
	f = open("madara_r_4_3.ply", "w")

	vertices = []
	v2index = {}
	index_counter = 0
	faces = []
	for t in triangles:
		out = []
		for vertex in t.vertices:
			v = tuple(vertex)
			if v2index.get(v):
				out.append(v2index[v])
			else:
				v2index[v] = index_counter
				index_counter += 1
				out.append(v2index[v])
				vertices.append(v)
		faces.append(out)

	f.write('ply\n')
	f.write('format ascii 1.0\n')
	f.write('element vertex ' + str(len(vertices)) + '\n')
	f.write('property float32 x\n')
	f.write('property float32 y\n')
	f.write('property float32 z\n')
	f.write('element face ' + str(len(faces)) + '\n')
	f.write('property list uint8 int32 vertex_indices\n')
	f.write('end_header\n')

	for v in vertices:
		f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
	for face in faces:
		f.write('3 ' + str(face[0]) + " " + str(face[1]) + " " + str(face[2]) + "\n")
	f.close()
