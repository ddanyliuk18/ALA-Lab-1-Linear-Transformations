import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

file_path = r"D:\Program files\PyCharm\ALA_1\Linear_Transformations\bookshelf\test\bookshelf_0573.off"

def load_off_with_faces(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    if lines[0] != 'OFF':
        raise ValueError("Not a valid OFF file")
    n_verts, n_faces, _ = map(int, lines[1].split())
    vertices = np.array([[float(v) for v in line.split()] for line in lines[2:2 + n_verts]])
    faces = []
    for line in lines[2 + n_verts:2 + n_verts + n_faces]:
        parts = list(map(int, line.split()))
        faces.append(parts[1:])
    return vertices, faces

def plot_off(vertices, faces, color='lightblue', alpha=0.8, ax=None, label=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    mesh = [[vertices[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(mesh, alpha=alpha, facecolor=color, edgecolor='k', linewidths=0.1)
    ax.add_collection3d(collection)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if label:
        ax.set_title(label)
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    return ax

def rotate_xy(x, a):
    matrix = x.copy()
    xy_rotation = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a), np.cos(a), 0],
        [0,0,1]])
    xy_rotation_result = matrix @ xy_rotation
    return xy_rotation_result

def rotate_yz(x, a):
    matrix = x.copy()
    yz_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a), np.cos(a)]])
    yz_rotation_result = matrix @ yz_rotation
    return yz_rotation_result

def rotate_zx(x, a):
    matrix = x.copy()
    zx_rotation = np.array([
        [np.cos(a), 0, -np.sin(a)],
        [0, 1, 0],
        [np.sin(a), 0, np.cos(a)]])
    zx_rotation_result = matrix @ zx_rotation
    return zx_rotation_result

vertices, faces = load_off_with_faces(file_path)
#rotated_vertices = rotate_xy(vertices, np.pi/6)
#rotated_vertices = rotate_yz(vertices, np.pi/6)
rotated_vertices = rotate_zx(vertices, np.pi/6)

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.8, ax=ax1, label='Original')
ax2 = fig.add_subplot(122, projection='3d')
plot_off(rotated_vertices, faces, color='orange', alpha=0.8, ax=ax2, label='Rotated XZ')
plt.show()
