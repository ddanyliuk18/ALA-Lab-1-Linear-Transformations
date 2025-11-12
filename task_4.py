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

def combined_matrix(order, angle):
    R_xy = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    R_yz = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]
    ])
    R_zx = np.array([
        [np.cos(angle), 0, -np.sin(angle)],
        [0, 1, 0],
        [np.sin(angle), 0, np.cos(angle)]
    ])

    matrices = {'XY': R_xy, 'YZ': R_yz, 'ZX': R_zx}
    M = np.eye(3)
    for key in order:
        M = M @ matrices[key]
    return M


vertices, faces = load_off_with_faces(file_path)
fig = plt.figure(figsize=(15, 5))

M1 = combined_matrix(['XY', 'YZ', 'ZX'], np.pi/6)
M2 = combined_matrix(['YZ', 'ZX', 'XY'], np.pi/6)
M3 = combined_matrix(['ZX', 'XY', 'YZ'], np.pi/6)
combo1 = vertices @ M1
combo2 = vertices @ M2
combo3 = vertices @ M3
ax1 = fig.add_subplot(131, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax1)
plot_off(combo1, faces, color='orange', alpha=0.9, ax=ax1, label='XY → YZ → ZX')
ax1.set_title('Order: XY → YZ → ZX')

ax2 = fig.add_subplot(132, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax2)
plot_off(combo2, faces, color='green', alpha=0.9, ax=ax2, label='YZ → ZX → XY')
ax2.set_title('Order: YZ → ZX → XY')

ax3 = fig.add_subplot(133, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax3)
plot_off(combo3, faces, color='purple', alpha=0.9, ax=ax3, label='ZX → XY → YZ')
ax3.set_title('Order: ZX → XY → YZ')

print("Matrix for XY → YZ → ZX:\n", M1, "\n")
print("Matrix for YZ → ZX → XY:\n", M2, "\n")
print("Matrix for ZX → XY → YZ:\n", M3, "\n")

plt.tight_layout()
plt.show()