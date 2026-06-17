

def noop_transform(face):
    return face

def rotate_face_90_degrees_ccw_around_z(face):
    # Rotate each point in the face 90 degrees counter-clockwise around the Z axis
    return [(y, -x, z) for x, z, y in face]

def rotate_face_90_degrees_ccw_around_x(face):
    # Rotate each point in the face 90 degrees counter-clockwise around the X axis
    return [(x, -z, y) for x, y, z in face]

def rotate_face_90_degrees_ccw_around_y(face):
    # Rotate each point in the face 90 degrees counter-clockwise around the Y axis
    return [(z, y, -x) for x, y, z in face]

