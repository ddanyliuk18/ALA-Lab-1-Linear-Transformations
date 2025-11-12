# ALA-Lab-1-Linear-Transformations
## In this README AI was used in otder to polish the report

# 🧮 Task 1
This task demonstrates the implementation of several fundamental **2D linear transformations** applied to a shape using **NumPy** and **Matplotlib**. The transformations include **stretching, shearing, reflection, and rotation**.

---

## 📋 Overview

Linear transformations are implemented as matrix operations that modify the coordinates of a 2D shape. Each transformation is applied by multiplying the coordinate matrix with a specific transformation matrix.

---

## ⚙️ Transformation Functions

The following functions implement core linear transformations:

```python
import numpy as np

def stretch(x, a, b):
    """
    Apply a stretching transformation to the input coordinates.
    
    Parameters:
        x (np.array): Input coordinate matrix (N x 2)
        a (float): Horizontal stretch factor
        b (float): Vertical stretch factor
    
    Returns:
        np.array: Transformed coordinates
    """
    lynx = x.copy()
    stretch_matrix = np.array([[a, 0], [0, b]])
    stretched_result = lynx @ stretch_matrix
    return stretched_result

def shear(x, a, b):
    """
    Apply a shearing transformation to the input coordinates.
    
    Parameters:
        x (np.array): Input coordinate matrix (N x 2)
        a (float): Horizontal shear factor
        b (float): Vertical shear factor
    
    Returns:
        np.array: Transformed coordinates
    """
    lynx = x.copy()
    shear_matrix = np.array([[1, a], [b, 1]])
    shear_result = lynx @ shear_matrix
    return shear_result

def reflection(x, a, b):
    """
    Apply a reflection transformation across a line defined by direction vector (a, b).
    
    Parameters:
        x (np.array): Input coordinate matrix (N x 2)
        a (float): X-component of the reflection line direction
        b (float): Y-component of the reflection line direction
    
    Returns:
        np.array: Transformed coordinates
    """
    lynx = x.copy()
    reflection_matrix = (1 / (a**2 + b**2)) * np.array([
        [a**2 - b**2, 2*a*b],
        [2*a*b, b**2 - a**2]
    ])
    reflection_result = lynx @ reflection_matrix
    return reflection_result

def rotation(x, a):
    """
    Apply a rotation transformation around the origin.
    
    Parameters:
        x (np.array): Input coordinate matrix (N x 2)
        a (float): Rotation angle in radians
    
    Returns:
        np.array: Transformed coordinates
    """
    lynx = x.copy()
    rotation_matrix = np.array([
        [np.cos(a), -np.sin(a)],
        [np.sin(a),  np.cos(a)]
    ])
    rotation_result = lynx @ rotation_matrix
    return rotation_result
```

---

## 📊 Visualization

The visualization displays four different transformations applied to the original shape, arranged in a 2×2 grid:

```python
import matplotlib.pyplot as plt

# Original shape coordinates
lynx = np.array([
    [209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44],
    [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42],
    [-21.30, 259.65], [-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58],
    [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03],
    [-168.05, 104.09], [-184.14, 66.67], [-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68],
    [-197.40, -29.24], [-182.25, -44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32],
    [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56], [-143.43, -287.72], [-161.42, -240.94],
    [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66],
    [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -362.57], [363.08, -302.92],
    [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13],
    [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33], [247.57, -359.06],
    [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16],
    [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
    [251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82],
    [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96], [209.70, 368.42]
])

# Create 2x2 subplot grid
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 1. Stretch Transformation
axes[0, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
stretched = stretch(lynx, 1.5, 0.7)
axes[0, 0].plot(stretched[:, 0], stretched[:, 1], label='stretched', color='orange')
axes[0, 0].set_title('Stretch (a=1.5, b=0.7)')
axes[0, 0].grid(True)
axes[0, 0].legend()

# 2. Shear Transformation
axes[0, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
sheared = shear(lynx, 0.5, 0)
axes[0, 1].plot(sheared[:, 0], sheared[:, 1], label='sheared', color='green')
axes[0, 1].set_title('Shear (a=0.5, b=0)')
axes[0, 1].grid(True)
axes[0, 1].legend()

# 3. Reflection Transformation
axes[1, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
reflected = reflection(lynx, 1, 0)
axes[1, 0].plot(reflected[:, 0], reflected[:, 1], label='reflected', color='red')
axes[1, 0].set_title('Reflection (a=1, b=0)')
axes[1, 0].grid(True)
axes[1, 0].legend()

# 4. Rotation Transformation
axes[1, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
rotated = rotation(lynx, np.pi / 3)
axes[1, 1].plot(rotated[:, 0], rotated[:, 1], label='rotated', color='purple')
axes[1, 1].set_title('Rotation (θ=π/3)')
axes[1, 1].grid(True)
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### Transformation Details:

- **Stretch**: Scales the shape horizontally by 1.5× and vertically by 0.7×
- **Shear**: Applies a horizontal shear with factor 0.5
- **Reflection**: Reflects the shape across the x-axis (horizontal line)
- **Rotation**: Rotates the shape by 60° (π/3 radians) counterclockwise

---

## 🖼️ Results
<img width="1536" height="754" alt="task1" src="https://github.com/user-attachments/assets/bae3d5f3-a8e8-4c40-a955-ff8c8e22ea46" />

---
# 🔄 Task 2 — Composition of Linear Transformations

This task explores how **the order of linear transformations affects the final result**. Multiple transformations (Stretch, Shear, Rotation, and Reflection) are applied in different sequences to demonstrate that **matrix multiplication is not commutative**.

---

## 📋 Overview

The key question addressed in this task is: **Does the final result depend on the order of transformations?**

The answer is **YES** — linear transformations are generally **non-commutative**, meaning that applying transformation A followed by B produces a different result than applying B followed by A.

---

## 🎯 Objective

Apply four transformations in different orders:
- **Stretch** (horizontal: 1.5×, vertical: 0.7×)
- **Shear** (horizontal factor: 0.5)
- **Rotation** (angle: π/4 or 45°)
- **Reflection** (across x-axis)

Each composition demonstrates how the sequence affects the final shape and position.

---

## 📊 Visualization Code

```python
import numpy as np
import matplotlib.pyplot as plt

# Create 2x2 subplot grid for different composition orders
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ============================================
# Composition 1: Stretch → Shear → Rotation → Reflection
# ============================================
axes[0, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='lavender')
stretched = stretch(lynx, 1.5, 0.7)
sheared = shear(stretched, 0.5, 0)
rotated = rotation(sheared, np.pi/4)
composition_1 = reflection(rotated, 1, 0)
axes[0, 0].plot(composition_1[:, 0], composition_1[:, 1], 
                label='Composition\n(Stretch + Shear + Rotation + Reflection)', 
                color='orchid')
axes[0, 0].set_title('Composition\n(Stretch + Shear + Rotation + Reflection)')
axes[0, 0].grid(True)
axes[0, 0].set_xlabel('X axis')
axes[0, 0].set_ylabel('Y axis')
axes[0, 0].legend()

# ============================================
# Composition 2: Stretch → Rotation → Shear → Reflection
# ============================================
axes[0, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='lavender')
stretched = stretch(lynx, 1.5, 0.7)
rotated = rotation(stretched, np.pi/4)
sheared = shear(rotated, 0.5, 0)
composition_2 = reflection(sheared, 1, 0)
axes[0, 1].plot(composition_2[:, 0], composition_2[:, 1], 
                label='Composition\n(Stretch + Rotation + Shear + Reflection)', 
                color='orchid')
axes[0, 1].set_title('Composition\n(Stretch + Rotation + Shear + Reflection)')
axes[0, 1].grid(True)
axes[0, 1].set_xlabel('X axis')
axes[0, 1].set_ylabel('Y axis')
axes[0, 1].legend()

# ============================================
# Composition 3: Reflection → Rotation → Stretch → Shear
# ============================================
axes[1, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='lavender')
reflected = reflection(lynx, 1, 0)
rotated = rotation(reflected, np.pi/4)
stretched = stretch(rotated, 1.5, 0.7)
composition_3 = shear(stretched, 0.5, 0)
axes[1, 0].plot(composition_3[:, 0], composition_3[:, 1], 
                label='Composition\n(Reflection + Rotation + Stretch + Shear)', 
                color='orchid')
axes[1, 0].set_title('Composition\n(Reflection + Rotation + Stretch + Shear)')
axes[1, 0].grid(True)
axes[1, 0].set_xlabel('X axis')
axes[1, 0].set_ylabel('Y axis')
axes[1, 0].legend()

# ============================================
# Composition 4: Shear → Reflection → Rotation → Stretch
# ============================================
axes[1, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='lavender')
sheared = shear(lynx, 0.5, 0)
reflected = reflection(sheared, 1, 0)
rotated = rotation(reflected, np.pi/4)
composition_4 = stretch(rotated, 1.5, 0.7)
axes[1, 1].plot(composition_4[:, 0], composition_4[:, 1], 
                label='Composition\n(Shear + Reflection + Rotation + Stretch)', 
                color='orchid')
axes[1, 1].set_title('Composition\n(Shear + Reflection + Rotation + Stretch)')
axes[1, 1].grid(True)
axes[1, 1].set_xlabel('X axis')
axes[1, 1].set_ylabel('Y axis')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

---

## 🔍 Composition Analysis

### Four Different Transformation Orders:

1. **Stretch → Shear → Rotation → Reflection**
   - First scales the shape, then skews it, rotates, and finally reflects

2. **Stretch → Rotation → Shear → Reflection**
   - Changes order: rotation happens before shearing
   - Results in a **different final shape and position**

3. **Reflection → Rotation → Stretch → Shear**
   - Starts with reflection, producing a mirrored base
   - Subsequent transformations apply to the reflected shape

4. **Shear → Reflection → Rotation → Stretch**
   - Shearing first affects how subsequent transformations behave
   - Creates yet another **distinct result**

---

## 📐 Mathematical Explanation

In matrix notation, composition of transformations is represented as:

```
Final = T₄ × T₃ × T₂ × T₁ × Original
```

Where each Tᵢ is a transformation matrix. Since **matrix multiplication is not commutative**:

```
T₁ × T₂ ≠ T₂ × T₁
```

This means **the order matters** — different sequences produce different results.

---

## 🖼️ Results

<img width="1536" height="754" alt="task2" src="https://github.com/user-attachments/assets/3edc69d8-e80c-49d1-b533-fb91cadad3b7" />

---
# 🌐 Task 3 — 3D Rotations

This task implements **three-dimensional rotation transformations** around different axes. A 3D model (loaded from an OFF file) is rotated around the XY, YZ, and ZX planes to demonstrate spatial transformations.

---

## 📋 Overview

In 3D space, rotations can occur around three primary planes:
- **XY-plane rotation** (rotation around Z-axis)
- **YZ-plane rotation** (rotation around X-axis)
- **ZX-plane rotation** (rotation around Y-axis)

Each rotation is represented by a specific 3×3 rotation matrix that preserves distances and angles while changing the orientation of the object.

---

## ⚙️ Rotation Functions

### 1. XY-Plane Rotation (Around Z-axis)

Rotates the object in the XY-plane, keeping the Z-coordinate unchanged.

```python
def rotate_xy(x, a):
    """
    Apply rotation in the XY-plane (around Z-axis).
    
    Parameters:
        x (np.array): Input vertex array (N x 3)
        a (float): Rotation angle in radians
    
    Returns:
        np.array: Rotated vertices
    """
    matrix = x.copy()
    xy_rotation = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])
    xy_rotation_result = matrix @ xy_rotation
    return xy_rotation_result
```

**Matrix explanation**:
- Affects X and Y coordinates
- Z-coordinate remains constant
- Rotates counterclockwise when viewed from positive Z-axis

---

### 2. YZ-Plane Rotation (Around X-axis)

Rotates the object in the YZ-plane, keeping the X-coordinate unchanged.

```python
def rotate_yz(x, a):
    """
    Apply rotation in the YZ-plane (around X-axis).
    
    Parameters:
        x (np.array): Input vertex array (N x 3)
        a (float): Rotation angle in radians
    
    Returns:
        np.array: Rotated vertices
    """
    matrix = x.copy()
    yz_rotation = np.array([
        [1,         0,          0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    yz_rotation_result = matrix @ yz_rotation
    return yz_rotation_result
```

**Matrix explanation**:
- X-coordinate remains constant
- Affects Y and Z coordinates
- Rotates counterclockwise when viewed from positive X-axis

---

### 3. ZX-Plane Rotation (Around Y-axis)

Rotates the object in the ZX-plane, keeping the Y-coordinate unchanged.

```python
def rotate_zx(x, a):
    """
    Apply rotation in the ZX-plane (around Y-axis).
    
    Parameters:
        x (np.array): Input vertex array (N x 3)
        a (float): Rotation angle in radians
    
    Returns:
        np.array: Rotated vertices
    """
    matrix = x.copy()
    zx_rotation = np.array([
        [np.cos(a),  0, -np.sin(a)],
        [0,          1,          0],
        [np.sin(a),  0,  np.cos(a)]
    ])
    zx_rotation_result = matrix @ zx_rotation
    return zx_rotation_result
```

**Matrix explanation**:
- Y-coordinate remains constant
- Affects Z and X coordinates
- Rotates counterclockwise when viewed from positive Y-axis

---

## 📦 Utility Functions

### Loading 3D Model from OFF File

```python
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

def load_off_with_faces(file_path):
    """
    Load a 3D model from an OFF (Object File Format) file.
    
    Parameters:
        file_path (str): Path to the OFF file
    
    Returns:
        tuple: (vertices, faces) - vertices as numpy array, faces as list of indices
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if lines[0] != 'OFF':
        raise ValueError("Not a valid OFF file")
    
    n_verts, n_faces, _ = map(int, lines[1].split())
    
    # Read vertices
    vertices = np.array([[float(v) for v in line.split()] 
                        for line in lines[2:2 + n_verts]])
    
    # Read faces
    faces = []
    for line in lines[2 + n_verts:2 + n_verts + n_faces]:
        parts = list(map(int, line.split()))
        faces.append(parts[1:])  # Skip the vertex count
    
    return vertices, faces
```

---

### 3D Visualization Function

```python
def plot_off(vertices, faces, color='lightblue', alpha=0.8, ax=None, label=None):
    """
    Plot a 3D mesh from vertices and faces.
    
    Parameters:
        vertices (np.array): Vertex coordinates (N x 3)
        faces (list): List of face indices
        color (str): Face color
        alpha (float): Transparency (0-1)
        ax: Matplotlib 3D axis (creates new if None)
        label (str): Plot title
    
    Returns:
        ax: The 3D axis object
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    
    # Create mesh from vertices and faces
    mesh = [[vertices[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(mesh, alpha=alpha, facecolor=color, 
                                 edgecolor='k', linewidths=0.1)
    ax.add_collection3d(collection)
    
    # Set labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    if label:
        ax.set_title(label)
    
    # Auto-scale axes
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    
    return ax
```

---

## 📊 Visualization Code

```python
# Load 3D model
file_path = "path/to/your/model.off"
vertices, faces = load_off_with_faces(file_path)

# Apply rotation (example: XY-plane rotation by 30°)
rotated_vertices = rotate_xy(vertices, np.pi/6)

# Create side-by-side comparison
fig = plt.figure(figsize=(10, 5))

# Original model
ax1 = fig.add_subplot(121, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.8, ax=ax1, label='Original')

# Rotated model
ax2 = fig.add_subplot(122, projection='3d')
plot_off(rotated_vertices, faces, color='orange', alpha=0.8, ax=ax2, label='Rotated XY')

plt.show()
```

---

## 🔍 Rotation Matrix Details

### General 3D Rotation Properties

All three rotation matrices share these properties:
- **Orthogonal**: R^T × R = I (transpose equals inverse)
- **Determinant = 1**: Preserves orientation and volume
- **Preserves distances**: ||R × v|| = ||v||
- **Preserves angles**: angle between R×v₁ and R×v₂ equals angle between v₁ and v₂

### Rotation Directions

When viewing from the **positive axis** toward the origin:
- **Positive angle**: Counterclockwise rotation
- **Negative angle**: Clockwise rotation

---

## 🖼️ Results

### XY-Plane Rotation (π/6 radians = 30°)
<img width="1000" height="500" alt="task3 1" src="https://github.com/user-attachments/assets/9be2e8e7-7f9f-4bb4-bf00-a0b5bcc995ca" />

### YZ-Plane Rotation (π/6 radians = 30°)
<img width="1000" height="500" alt="task3 2" src="https://github.com/user-attachments/assets/deeb96c5-2b64-49ab-9e11-0ad612692ab5" />

### ZX-Plane Rotation (π/6 radians = 30°)
<img width="1000" height="500" alt="task3 3" src="https://github.com/user-attachments/assets/730d8902-871c-41f9-88d3-1fb970f002c9" />
---

# 🔢 Task 4 — Combined Rotation Matrix

This task demonstrates how to **compute the combined transformation matrix** for multiple 3D rotations applied in sequence. Instead of applying rotations step-by-step, we calculate a single composite matrix that represents the entire transformation.

---

## 📋 Overview

When multiple rotations are applied sequentially, the final transformation can be represented by a **single combined matrix**. This approach is more efficient computationally and provides insight into the mathematical structure of composed rotations.

**Key Concept**: Instead of:
```python
result = rotate_zx(rotate_yz(rotate_xy(vertices, angle), angle), angle)
```

We compute:
```python
M_combined = R_ZX @ R_YZ @ R_XY
result = vertices @ M_combined
```

---

## ⚙️ Combined Matrix Function

```python
def combined_matrix(order, angle):
    """
    Compute the combined rotation matrix for a sequence of 3D rotations.
    
    Parameters:
        order (list): List of rotation plane identifiers in order ['XY', 'YZ', 'ZX']
        angle (float): Rotation angle in radians (same for all rotations)
    
    Returns:
        np.array: Combined 3x3 transformation matrix
    """
    # Define individual rotation matrices
    R_xy = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0,              0,              1]
    ])
    
    R_yz = np.array([
        [1,             0,               0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])
    
    R_zx = np.array([
        [np.cos(angle),  0, -np.sin(angle)],
        [0,              1,              0],
        [np.sin(angle),  0,  np.cos(angle)]
    ])
    
    # Map string identifiers to matrices
    matrices = {'XY': R_xy, 'YZ': R_yz, 'ZX': R_zx}
    
    # Start with identity matrix
    M = np.eye(3)
    
    # Multiply matrices in the specified order
    for key in order:
        M = M @ matrices[key]
    
    return M
```

---

## 📊 Visualization with Combined Matrices

```python
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# Load 3D model
file_path = "path/to/your/model.off"
vertices, faces = load_off_with_faces(file_path)

# ============================================
# Compute combined matrices for three different orders
# ============================================
M1 = combined_matrix(['XY', 'YZ', 'ZX'], np.pi/6)
M2 = combined_matrix(['YZ', 'ZX', 'XY'], np.pi/6)
M3 = combined_matrix(['ZX', 'XY', 'YZ'], np.pi/6)

# Apply transformations using matrix multiplication
combo1 = vertices @ M1
combo2 = vertices @ M2
combo3 = vertices @ M3

# ============================================
# Visualization
# ============================================
fig = plt.figure(figsize=(15, 5))

# Combination 1: XY → YZ → ZX
ax1 = fig.add_subplot(131, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax1)
plot_off(combo1, faces, color='orange', alpha=0.9, ax=ax1, label='XY → YZ → ZX')
ax1.set_title('Order: XY → YZ → ZX')

# Combination 2: YZ → ZX → XY
ax2 = fig.add_subplot(132, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax2)
plot_off(combo2, faces, color='green', alpha=0.9, ax=ax2, label='YZ → ZX → XY')
ax2.set_title('Order: YZ → ZX → XY')

# Combination 3: ZX → XY → YZ
ax3 = fig.add_subplot(133, projection='3d')
plot_off(vertices, faces, color='lightblue', alpha=0.4, ax=ax3)
plot_off(combo3, faces, color='purple', alpha=0.9, ax=ax3, label='ZX → XY → YZ')
ax3.set_title('Order: ZX → XY → YZ')

# ============================================
# Print transformation matrices
# ============================================
print("Matrix for XY → YZ → ZX:\n", M1, "\n")
print("Matrix for YZ → ZX → XY:\n", M2, "\n")
print("Matrix for ZX → XY → YZ:\n", M3, "\n")

plt.tight_layout()
plt.show()
```

---

## 🔍 Transformation Matrices Output

### Combination 1: XY → YZ → ZX

```
Matrix M1 = R_ZX × R_YZ × R_XY:

[[ 0.875      -0.4330127  -0.21650635]
 [ 0.21650635  0.75       -0.625     ]
 [ 0.4330127   0.5         0.75      ]]
```

**Interpretation**: This matrix encodes the complete transformation when first rotating in the XY-plane, then YZ-plane, then ZX-plane (each by π/6 radians).

---

### Combination 2: YZ → ZX → XY

```
Matrix M2 = R_XY × R_ZX × R_YZ:

[[ 0.75       -0.4330127  -0.5       ]
 [ 0.21650635  0.875      -0.4330127 ]
 [ 0.625       0.21650635  0.75      ]]
```

**Interpretation**: Different order produces a **different combined matrix**, resulting in a different final orientation.

---

### Combination 3: ZX → XY → YZ

```
Matrix M3 = R_YZ × R_XY × R_ZX:

[[ 0.75       -0.625      -0.21650635]
 [ 0.5         0.75       -0.4330127 ]
 [ 0.4330127   0.21650635  0.875     ]]
```

**Interpretation**: Yet another order yields another unique transformation matrix.

---

## 📐 Matrix Properties Analysis

All three combined matrices share these properties of rotation matrices:

### 1. **Orthogonality**
```python
# For any rotation matrix R:
# R^T × R = I (Identity matrix)
# This means: R^(-1) = R^T
```

### 2. **Determinant = 1**
```python
det(M1) = det(M2) = det(M3) = 1.0
```
This preserves orientation and volume.

### 3. **Preservation of Lengths**
```python
# For any vector v:
# ||M × v|| = ||v||
```
Distances are preserved under rotation.

### 4. **Non-Commutativity**
```
M1 ≠ M2 ≠ M3
```
Despite using the same three rotations, different orders produce different matrices!

---

## 🖼️ Results

### Combined Rotations Comparison

<img width="1536" height="754" alt="task4" src="https://github.com/user-attachments/assets/dead229a-0039-4771-b787-cbee1220c87c" />

*The visualization shows three different final orientations from the same rotations applied in different orders, along with their corresponding transformation matrices*

---
