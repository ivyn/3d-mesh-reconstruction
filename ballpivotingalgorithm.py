import numpy as np
import open3d as o3d
import sys


class Vertex:
	def __init__(self, v):
		self.x = v[0]
		self.y = v[1]
		self.z = v[2]
		self.edges = set()

	def __eq__(self, other):
		return other.x == self.x and other.y == self.y and other.z == self.z

	def __str__(self):
		return str(self.x) + str(self.y) + str(self.z)

	def add_edge(self, e):
		self.edges.add(e)


class Edge:
	def __init__(self, v1, v2, opp):
		self.v1 = v1
		self.v2 = v2
		self.triangles = 1
		self.on_front = True
		self.opposite = opp

	def __hash__(self):
		v1 = self.v1
		v2 = self.v2
		if v1[0] < v2[0] or (v1[0] == v2[0] and v1[1] < v2[1]) or (v1[0] == v2[0] and v1[1] == v2[1] and v1[2] < v2[2]):
			return self.get_str(v1, v2)
		else:
			return self.get_str(v2, v1)

	def get_str(self, v1, v2):
		return hash(str(v1[0]) + str(v1[1]) + str(v1[2]) + str(v2[0]) + str(v2[1]) + str(v2[2]))


class Triangle:
	def __init__(self, v1, v2, v3):
		self.v1 = v1
		self.v2 = v2
		self.v3 = v3
		self.vertices = [v1, v2, v3]

	def __eq__(self, other):
		vertices = [self.v1, self.v2, self.v3]
		return other.v1 in vertices and other.v2 in vertices and other.v3 in vertices


class Front:
	def __init__(self):
		self.active_edges = set()
		self.active_vertices = set()

	def add_edge(self, edge):
		self.active_edges.add(edge)

	def remove_edge(self, edge):
		self.active_edges.remove(edge)

	def __contains__(self, edge):
		return edge in self.active_edges

	def isEmpty(self):
		return not len(self.active_edges)


class BPA:
	def __init__(self, point_cloud):
		self.pc = point_cloud
		self.find_rad()
		self.construct_voxel()
		self.seen = set()
		self.used_vertices = dict()
		self.front = Front()
		self.triangle_list = []

	def find_rad(self):
		self.radius = np.average(self.pc.compute_nearest_neighbor_distance()) / 1.5 * 2

	def construct_voxel(self):
		box_pts = np.asarray(self.pc.get_axis_aligned_bounding_box().get_box_points())
		self.x_min = np.min(box_pts[:,0])
		self.y_min = np.min(box_pts[:,1])
		self.z_min = np.min(box_pts[:,2])
		x_max = np.max(box_pts[:,0])
		y_max = np.max(box_pts[:,1])
		z_max = np.max(box_pts[:,2])

		self.x_width = int((x_max - self.x_min) / self.radius + 1)
		self.y_width = int((y_max - self.y_min) / self.radius + 1)
		self.z_width = int((z_max - self.z_min) / self.radius + 1)

		self.voxel = [[[[] for i in range(self.z_width)] for j in range(self.y_width)] for k in range(self.x_width)]

		r = self.radius

		for pt, n in zip(np.asarray(self.pc.points), np.asarray(self.pc.normals)):
			x = int((pt[0] - self.x_min) / r)
			y = int((pt[1] - self.y_min) / r)
			z = int((pt[2] - self.z_min) / r)
			self.voxel[x][y][z].append([pt, n])

	def get_neighbors(self, pt, rad):
		# Checks all 27 voxels surrounding the point's voxel
		# Returns an array of neighbors that are within 2r of that point
		result = []
		voxel_dist = int(rad / self.radius)
		x = int((pt[0] - self.x_min) / self.radius)
		y = int((pt[1] - self.y_min) / self.radius)
		z = int((pt[2] - self.z_min) / self.radius)

		for i in range(max(0, x - voxel_dist), min(x + voxel_dist + 1, self.x_width)):
			for j in range(max(0, y - voxel_dist), min(y + voxel_dist + 1, self.y_width)):
				for k in range(max(0, z - voxel_dist), min(z + voxel_dist + 1, self.z_width)):
					voxel_elems = self.voxel[i][j][k]
					for neighbor in voxel_elems:
						distance = np.linalg.norm(pt - neighbor[0])
						if distance <= rad + sys.float_info.min and distance > 0:
							result.append(neighbor)
		return result

	def get_ball_centers(self, p1, p2, p3, r):
		p21 = p2 - p1
		p31 = p3 - p1
		n = np.cross(p21,p31)

		# If points are collinear
		if not n.any():
			return np.array([]), None, None

		# p0 = center of circumscribing circle
		# min_radius = distance between p0 and point on triangle
		numerator_left = np.cross(np.linalg.norm(p3 - p1) ** 2 * (np.cross(p2 - p1, p3 - p1)), (p2 - p1))
		numerator_right = np.cross(np.linalg.norm(p2 - p1) ** 2 * (np.cross(p3 - p1, p2 - p1)), (p3 - p1))
		denominator = (2 * np.linalg.norm(np.cross(p2 - p1, p3 - p1)) ** 2)
		p0 = p1 + (numerator_left + numerator_right) / denominator
		min_radius = np.linalg.norm(p1 - p0)

		if r >= min_radius:
			theta = np.arccos(min_radius / self.radius)
			t = np.sin(theta) * self.radius
			# magnitude = sqrt(n[0]**2 + n[1]**2 + n[2]**2)
			unit_vector = n / np.linalg.norm(n)
			center1 = p0 + unit_vector * t
			center2 = p0 + unit_vector * -t
			return center1, center2, unit_vector
		else:
			return np.array([]), None, None

	def validate_ball_center(self, center, unit_vec, pn):
		# Check if there are no closer points to the center
		# Check if the normal and the center are on the same side of the plane defined by the 3 points
		neighbors_len = len(self.get_neighbors(center, self.radius))
		if neighbors_len == 3:
			angle = np.arccos(np.dot(unit_vec, pn[1] / np.linalg.norm(pn[1])))
			if angle <= 1.5707963267948966:
				return True
			return False

	def find_seed_triangle(self):
		"""
		To find the seed triangle, we iterate through the points and check if the point has two neighbors within a 2p distance
		of that point. If there is a set of three points all within a 2p distance of each other, we then check if the triangle
		normal is consistent with the vertex normals (and adjust if it is not).
		We also check if the triangle is contained in a P-Ball.
		If these requirements are met, we return a seed triangle, if not we keep searching through
		the points to find another possible seed triangle.
		"""
		for pt, norm in zip(np.asarray(self.pc.points), np.asarray(self.pc.normals)): #TODO: change to pts not yet used

			if str(Vertex(pt)) in self.used_vertices:
				continue
			neighbors = self.get_neighbors(pt, 2 * self.radius)

			for i in range(len(neighbors) - 1):
				for j in range(i + 1, len(neighbors)):
					pt_1 = neighbors[i][0]
					pt_2 = neighbors[j][0]

					center1, center2, unit_vect = self.get_ball_centers(pt, pt_1, pt_2, self.radius)
					triangle = None
					if len(center1):
						if self.validate_ball_center(center1, unit_vect, [pt, norm]):
							triangle = Triangle(pt, pt_1, pt_2)
						elif self.validate_ball_center(center2, -unit_vect, [pt, norm]):
							triangle = Triangle(pt, pt_1, pt_2)
						if triangle:
							v1 = Vertex(pt)
							v2 = Vertex(pt_1)
							v3 = Vertex(pt_2)
							e1 = Edge(pt, pt_1, pt_2)
							e2 = Edge(pt_1, pt_2, pt)
							e3 = Edge(pt_2, pt, pt_1)
							self.used_vertices[str(v1)] = v1
							self.used_vertices[str(v2)] = v2
							self.used_vertices[str(v3)] = v3
							v1.edges.update([e1, e3])
							v2.edges.update([e1, e2])
							v3.edges.update([e2, e3])
							self.front.add_edge(e1)
							self.front.add_edge(e2)
							self.front.add_edge(e3)
							return triangle
			self.used_vertices[str(Vertex(pt))] = Vertex(pt)
		return None

	def pivot_ball(self, e):
		# If the distance to the plane < the ball radius, then intersection
		# between a ball centered in the point and the plane exists
		midpoint = (e.v1 + e.v2) / 2

		candidates = self.get_neighbors(midpoint, 2 * self.radius)

		for c in candidates:
			if (c[0] == e.opposite).all():
				continue

			center1, center2, unit_vect = self.get_ball_centers(e.v1, e.v2, c[0], self.radius)
			if len(center1) > 0:
				if self.validate_ball_center(center1, unit_vect, c):
					triangle = Triangle(e.v1, e.v2, c[0])
				elif self.validate_ball_center(center2, -unit_vect, c):
					triangle = Triangle(e.v1, e.v2, c[0])
			else:
				continue
		  # If c is not used, then join
			if self.used_vertices.get(str(Vertex(c[0]))) is None:
				self.used_vertices[str(Vertex(c[0]))] = Vertex(c[0])
				e.on_front = False
				e.triangles = 2
				e1 = Edge(c[0], e.v1, e.v2)
				e2 = Edge(c[0], e.v2, e.v1)
				e1.triangles = 1
				self.used_vertices[str(Vertex(e.v1))].edges.add(e1)
				self.used_vertices[str(Vertex(e.v2))].edges.add(e2)
				self.front.add_edge(e1)
				self.front.add_edge(e2)
				self.front.remove_edge(e)
				return Triangle(c[0], e.v1, e.v2)

		  # If c is used
			else:
			  # C is part of active edges, glued together
				e1 = Edge(c[0], e.v1, e.v2)
				e2 = Edge(c[0], e.v2, e.v1)
				if e1 in self.front or e2 in self.front:
					if e1 in self.front:
						self.front.remove_edge(e1)
					else:
						self.front.add_edge(e1)
					if e2 in self.front:
						self.front.remove_edge(e2)
					else:
						self.front.add_edge(e2)
					self.front.remove_edge(e)
					return Triangle(c[0], e.v1, e.v2)
		self.front.remove_edge(e)

	def make_mesh(self):
		# Iterates until all vertices have been considered
		num_tris = 0
		while True:
			seed = self.find_seed_triangle()

			if not seed:
				break
			else:
				num_tris += 1

			while not self.front.isEmpty():
				e = self.front.active_edges.pop()
				self.front.add_edge(e)
				triangle = self.pivot_ball(e)
				if triangle:
					num_tris += 1
					if num_tris % 1000 == 0:
						print(num_tris)
					self.triangle_list.append(triangle)
		return self.triangle_list
