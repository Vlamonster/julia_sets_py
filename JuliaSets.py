import math
import time
from tkinter import *
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from functools import partial

root = Tk()
root.title("Julia Sets")

MAX_STEPS = 500


def get_coordinate_matrix(width, height):
    indices_x = np.linspace(start=-2, stop=2, num=width, endpoint=True)
    indices_y = np.linspace(start=-2 * height / width, stop=2 * height / width, num=height, endpoint=True)

    indices_y, indices_x = np.meshgrid(indices_y, indices_x)

    return indices_x + 1j * indices_y


def get_steps_matrix(coordinate_matrix, c):
    width, height = coordinate_matrix.shape
    steps_matrix = np.zeros(shape=(width, height))
    absolute_matrix = np.zeros(shape=(width, height))
    escaped_matrix = np.zeros(shape=(width, height), dtype=bool)
    not_escaped_matrix = np.ones(shape=(width, height), dtype=bool)
    for step in range(MAX_STEPS):
        np.abs(coordinate_matrix, where=not_escaped_matrix, out=absolute_matrix)

        np.greater(absolute_matrix, 4, where=not_escaped_matrix, out=escaped_matrix)
        np.logical_not(escaped_matrix, out=not_escaped_matrix)

        np.add(steps_matrix, 1, where=not_escaped_matrix, out=steps_matrix)

        np.square(coordinate_matrix, where=not_escaped_matrix, out=coordinate_matrix)
        np.add(coordinate_matrix, c, where=not_escaped_matrix, out=coordinate_matrix)

    return steps_matrix + (12 - (absolute_matrix - 4)) / 12


def get_color_matrix(steps_matrix):
    width, height = steps_matrix.shape
    X = np.transpose(steps_matrix)
    X = np.flip(X, axis=1)
    color_matrix = np.zeros(shape=(height, width), dtype=float)

    color_matrix[:, :] = np.where(X == MAX_STEPS, 0, np.mod(15 * X, MAX_STEPS) / MAX_STEPS)

    cm = plt.get_cmap("inferno")

    return cm(color_matrix, bytes=True)


start = time.time()

A = get_coordinate_matrix(1920, 1280)
get_steps = partial(get_steps_matrix, c=-0 + 1j)

with ThreadPool(12) as f:
    B_slices = f.map(get_steps, np.array_split(A, 12))

B = np.concatenate(B_slices)
C = get_color_matrix(B)

end = time.time()
print(end - start)

image = ImageTk.PhotoImage(Image.fromarray(C, mode="RGBA"))
label = Label(root, image=image)
label.pack()

root.mainloop()
