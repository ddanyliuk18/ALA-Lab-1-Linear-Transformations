import numpy as np
import matplotlib.pyplot as plt

lynx = np.array([
[209.70, 368.42], [157.63, 332.16], [118.82, 284.21], [80.95, 224.56], [43.08, 244.44], [20.36, 266.67], [-4.26, 293.57], [2.37, 263.16], [-20.36, 292.40], [-39.29, 299.42], [-21.30, 259.65],
[-50.65, 267.84], [-39.29, 242.11], [-55.38, 240.94], [-100.83, 300.58], [-149.11, 345.03], [-172.78, 361.40], [-189.82, 300.58], [-192.66, 225.73], [-181.30, 145.03], [-168.05, 104.09], [-184.14, 66.67],
[-186.98, 31.58], [-183.20, 3.51], [-208.76, -4.68], [-197.40, -29.24], [-182.25, -44.44], [-203.08, -43.27], [-172.78, -92.40], [-131.12, -126.32], [-101.78, -147.37], [-74.32, -163.74], [-110.30, -224.56],
[-143.43, -287.72], [-161.42, -240.94], [-282.60, -221.05], [-388.64, -205.85], [-370.65, -301.75], [-339.41, -397.66], [18.46, -397.66], [345.09, -400.00], [359.29, -378.95], [367.81, -342.69], [346.98, -
362.57], [363.08, -302.92], [357.40, -243.27], [348.88, -266.67], [336.57, -201.17], [290.18, -135.67], [240.00, -118.13], [258.93, -164.91], [257.99, -228.07], [252.31, -271.35], [256.09, -333.33],
[247.57, -359.06], [230.53, -307.60], [194.56, -238.60], [160.47, -181.29], [120.71, -149.71], [165.21, -132.16], [201.18, -100.58], [183.20, -99.42], [221.07, -73.68], [253.25, -24.56], [222.01, -23.39],
[251.36, -1.17], [262.72, 24.56], [234.32, 25.73], [214.44, 42.11], [202.13, 60.82], [220.12, 101.75], [234.32, 160.23], [240.00, 230.41], [232.43, 316.96], [209.70, 368.42]
])
def stretch (x,a, b):
    lynx = x.copy()
    stretch_matrix = np.array([[a, 0], [0, b]])
    stretched_result = lynx @ stretch_matrix
    return stretched_result

def shear (x, a, b):
    lynx = x.copy()
    shear_matrix = np.array ([[1, a], [b, 1]])
    shear_result = lynx @ shear_matrix
    return shear_result

def reflection (x, a, b):
    lynx = x.copy()
    reflection_matrix = (1/ (a**2 + b**2)) * np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])
    reflection_result = lynx @ reflection_matrix
    return reflection_result

def rotation(x, a):
    lynx = x.copy()
    rotation_matrix = np.array ([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] )
    rotation_result = lynx @ rotation_matrix
    return rotation_result

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
stretched = stretch(lynx, 1.5, 0.7)
axes[0, 0].plot(stretched[:, 0], stretched[:, 1], label='stretched', color='orange')
axes[0, 0].set_title('Stretch (a=1.5, b=0.7)')
axes[0, 0].grid(True)
axes[0, 0].set_xlabel('X axis')
axes[0, 0].set_ylabel('Y axis')
axes[0, 0].legend()

axes[0, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
sheared = shear(lynx, 0.5, 0)
axes[0, 1].plot(sheared[:, 0], sheared[:, 1], label='sheared', color='green')
axes[0, 1].set_title('Shear (a=0.5, b=0)')
axes[0, 1].grid(True)
axes[0, 1].set_xlabel('X axis')
axes[0, 1].set_ylabel('Y axis')
axes[0, 1].legend()

axes[1, 0].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
reflected = reflection(lynx, 1, 0)
axes[1, 0].plot(reflected[:, 0], reflected[:, 1], label='reflected', color='red')
axes[1, 0].set_title('Reflection (a=1, b=0)')
axes[1, 0].grid(True)
axes[1, 0].set_xlabel('X axis')
axes[1, 0].set_ylabel('Y axis')
axes[1, 0].legend()

axes[1, 1].plot(lynx[:, 0], lynx[:, 1], label='original', color='grey')
rotated = rotation(lynx, np.pi / 3)
axes[1, 1].plot(rotated[:, 0], rotated[:, 1], label='rotated', color='purple')
axes[1, 1].set_title('Rotation (θ=π/3)')
axes[1, 1].grid(True)
axes[1, 1].set_xlabel('X axis')
axes[1, 1].set_ylabel('Y axis')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
