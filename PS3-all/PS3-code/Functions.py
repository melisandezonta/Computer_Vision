import csv
import numpy as np
import cv2

def extract_2D_points(file):

    array_coordinates = []
    with open(file) as f:
        for line in csv.reader(f, delimiter=' '):
             line = list(filter(None, line))
             coords = [float(line[0]),float(line[1])]
             array_coordinates.append(coords)
    return np.asarray(array_coordinates)

def extract_3D_points(file):

    array_coordinates = []
    with open(file) as f:
        for line in csv.reader(f, delimiter=' '):
             line = list(filter(None, line))
             coords = [float(line[0]),float(line[1]),float(line[2])]
             array_coordinates.append(coords)
    return np.asarray(array_coordinates)


def conv_2_homogeneous_coords(array): return np.insert(array,array.shape[1] , 1, axis=1)


def residuals_calculation(points_2D,points_3D,M):
    residuals = np.zeros(points_2D.shape[0])
    for i in range(residuals.shape[0]):
        y = M.dot(points_3D[i,:])
        y = y / y[2]
        residuals[i] = np.sqrt(np.sum(np.power(points_2D[i,:] - y, 2)))
    return residuals


def svd_solver(points_2D,points_3D):

    u = points_2D[:,0]
    v = points_2D[:,1]
    X = points_3D[:,0]
    Y = points_3D[:,1]
    Z = points_3D[:,2]

    A = np.zeros(((points_3D.shape[0])*2,12), dtype=np.float32)

    ones_vector = np.ones(points_2D.shape[0])
    zeros_vectors = np.zeros(((points_2D.shape[0]),4), dtype=np.float32)

    A[::2,:] = np.column_stack((X, Y, Z, ones_vector,
                                  zeros_vectors, -u*X, -u*Y, -u*Z, -u))
    A[1::2,:] = np.column_stack((zeros_vectors, X, Y, Z,
                                         ones_vector, -v*X, -v*Y, -v*Z, -v))

    U,S,V = np.linalg.svd(A)

    M = V.T[:, -1]
    M = M.reshape((3,4))

    residuals = residuals_calculation(points_2D, points_3D, M)

    return M, residuals

def camera_center(M): return np.dot(-np.linalg.inv( M[:,:3]), M[:,3])

def svd_fundamental(points_2D_a, points_2D_b):

    u = points_2D_a[:,0]
    v = points_2D_a[:,1]
    u_p = points_2D_b[:,0]
    v_p = points_2D_b[:,1]


    A = np.column_stack((u*u_p, v*u_p, u_p, u*v_p, v*v_p, v_p, u, v, np.ones(points_2D_a.shape[0])))

    U,S,V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:,-1]
    F = F.reshape((3,3))
    return F


def epipolar_lines(image_a,image_b, array_2D_points, F):

    (rows,cols,_) = image_a.shape

    # Ll is defined as the line corresponding to the left hand side of the image
    # Lr is defined as the line corresponding to the right hand side of the image

    Ll = np.cross([0,0,1],[0,rows,1])
    Lr = np.cross([cols,0,1],[cols,rows,1])

    for i in range(array_2D_points.shape[0]):

        Li = F.dot(array_2D_points[i,:])
        Pil = np.cross(Li, Ll)
        Pil = Pil/Pil[-1]
        Pil = tuple(np.round(Pil[:-1]).astype(int))

        Pir = np.cross(Li, Lr)
        Pir = Pir/Pir[-1]
        Pir = tuple(np.round(Pir[:-1]).astype(int))


        cv2.line(image_a, Pil, Pir, (255,0,0), 1)

        markers = tuple(np.asarray(array_2D_points[i,:-1], dtype="int"))
        cv2.circle(image_b, markers, 2, (0,0,255), -1)




