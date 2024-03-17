import numpy as np
import uuid
from functools import lru_cache

from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import Vec3, LQuaternionf
from panda3d.bullet import BulletGenericConstraint, BulletRigidBodyNode
from panda3d.core import TransformState, NodePath
from panda3d.bullet import BulletWorld

from constants import material_properties, VoxelType, voxel_type_map
from utils import index_to_voxel_grid_coordinates
from geom import create_mesh


class DynamicArbitraryVoxelObject:

    def __init__(self, 
                 voxel_array: np.ndarray, 
                 voxel_size: float, 
                 vertices: np.ndarray, 
                 indices: np.ndarray, 
                 mass: float, 
                 friction: float, 
                 name: str = "VoxelObject"):
        
        #self.root = self
        #self.parent = self.root
        #self.parent_position_relative_to_grid = Vec3(0, 0, 0)

        self.voxel_array = voxel_array
        self.voxel_size = voxel_size
        self.vertices = vertices
        self.indices = indices
        
        self.name = name
        self.id = str(uuid.uuid4())
        self.mass = mass
        self.friction = friction
        
        self.node_paths: dict[tuple[int, int, int], NodePath] = {}

    def get_offset(self) -> np.ndarray: # deprecated?
        return (np.array(self.voxel_array.shape) - 1) // 2

    def add_voxel(self, hit_pos: Vec3, hit_normal: Vec3, voxel_type: VoxelType, game_engine):
        root_node_path = self.node_paths[(0, 0, 0)]
        root_node_pos = root_node_path.getPos()       
        orientation = root_node_path.getQuat()

        # Use the quaternion to transform the local forward direction to world space
        world_forward = Vec3(0, 1, 0)
        local_forward = orientation.xform(world_forward)

        ix = 1# TODO
        iy = 1# TODO
        iz = 1# TODO


        max_i, max_j, max_k = self.voxel_array.shape
        if not (0 <= ix < max_i and 0 <= iy < max_j and 0 <= iz < max_k):
            self.extend_array_uniformly()
        self.voxel_array[ix, iy, iz] = voxel_type.value

        self.vertices, self.indices = create_object_mesh(self)

        body_indices = np.argwhere(self.voxel_array)
        parts = []
        for i, j, k in body_indices:
            ix, iy, iz = index_to_voxel_grid_coordinates(i, j, k, self.voxel_array.shape[0])

            position = root_node_pos

            node_np = create_dynamic_single_voxel_physics_node(self, game_engine.render, game_engine.physics_world)
            node_np.setPythonTag("object", self)
            node_np.setQuat(orientation)
            node_np.setPos(position)
            self.node_paths[(i, j, k)] = node_np
            parts.append(node_np)

        for i in range(1, len(parts)):
            # Assuming bodyA and bodyB are instances of BulletRigidBodyNode you want to connect
            constraint = BulletGenericConstraint(parts[i-1].node(), parts[i].node(), TransformState.makeIdentity(), TransformState.makeIdentity(), True)

            # Lock all degrees of freedom to simulate a fixed constraint
            for i in range(6):
                constraint.setParam(i, 0, 0) 
            game_engine.physics_world.attachConstraint(constraint)

        velocity = root_node_path.node().getLinearVelocity()

        return root_node_pos, velocity, orientation

    def extend_array_uniformly(self):
        # Specify the padding width of 1 for all sides of all dimensions
        pad_width = [(1, 1)] * 3  # Padding for depth, rows, and columns
        
        # Pad the array with 0's on all sides
        self.voxel_array = np.pad(self.voxel_array, pad_width=pad_width, mode='constant', constant_values=0)

    def get_pos(self, render):
        # TODO: somehow calculate position - center of all the nodes? origin voxel?
        # To get the position relative to the parent node
        origin_node = self.node_paths[(0, 0)]
        local_position = origin_node.getPos()

        # To get the global position (relative to the world or render node)
        global_position = origin_node.getPos(render)

        print(f"Local Position: {local_position}")
        print(f"Global Position: {global_position}")

    def get_orientation(self) -> LQuaternionf:
        root_node_path = self.node_paths[(0, 0, 0)]
        return root_node_path.getQuat()
    
    def set_velocity(self, velocity: Vec3):
        for node_np in self.node_paths.values():
            node = node_np.node()
            node.setLinearVelocity(velocity)

    def enable_ccd(self):
        for node_np in self.node_paths.values():
            node = node_np.node()
            #voxel_diagonal = math.sqrt(3 * self.voxel_size**2)
            ccd_radius = self.voxel_size / 2 #voxel_diagonal / 2
            node.setCcdMotionThreshold(1e-7)
            node.setCcdSweptSphereRadius(ccd_radius)


def create_dynamic_single_voxel_object(voxel_size: int, voxel_type: VoxelType, render, physics_world) -> DynamicArbitraryVoxelObject:
    voxel_array = np.ones((1, 1, 1), dtype=int)
    vertices, indices = create_single_voxel_mesh(voxel_type, voxel_size)

    mass = material_properties[voxel_type]["mass"]
    friction = material_properties[voxel_type]["friction"]
    
    object = DynamicArbitraryVoxelObject(voxel_array, voxel_size, vertices, indices, mass, friction)
    object_np = create_dynamic_single_voxel_physics_node(object, render, physics_world)
    object_np.setPythonTag("object", object)
    object_np.setPythonTag("ijk", (0, 0, 0))
    object.node_paths[(0, 0, 0)]  = object_np
    return object            


def create_dynamic_single_voxel_physics_node(object: DynamicArbitraryVoxelObject, render: NodePath, physics_world: BulletWorld) -> NodePath:
    voxel_type_value = object.voxel_array[0, 0, 0]
    voxel_type = voxel_type_map[voxel_type_value]
    material = material_properties[voxel_type]
    
    radius = object.voxel_size / 2
    shape = BulletSphereShape(radius)
    node = BulletRigidBodyNode(object.name)
    node.setMass(material["mass"])
    node.setFriction(material["friction"])
    node.addShape(shape)

    # Set the initial position of the segment
    node_np = render.attachNewNode(node)
    
    # Add the node to the Bullet world
    physics_world.attachRigidBody(node)
    return node_np

def create_single_voxel_mesh(voxel_type: VoxelType, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    voxel_array = np.zeros((1, 1, 1), np.uint8)
    voxel_array[0, 0, 0] = voxel_type.value
    return create_mesh(voxel_array, voxel_array, voxel_size)

def create_object_mesh(object: DynamicArbitraryVoxelObject) -> tuple[np.ndarray, np.ndarray]:
    exposed_voxels = identify_exposed_voxels(object.voxel_array)
    return create_mesh(object.voxel_array, exposed_voxels, object.voxel_size)

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

def identify_exposed_voxels(voxel_array: np.ndarray) -> np.ndarray:
    """
    Identifies a voxel exposed to air and returns a same shaped boolean np array with the result.
    True means it is exposed to air, False means it's not.

    Parameters:
        - world_array: a 3D numpy array representing the voxel types as integers in the world
    """
    # Pad the voxel world with zeros (air) on all sides
    padded_world = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=0)
    
    # Initialize a boolean array for exposed faces
    exposed_faces = np.zeros_like(voxel_array, dtype=bool)
    
    # Check all six directions in a vectorized manner
    exposed_faces |= ((padded_world[:-2, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[2:, 1:-1, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, :-2, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 2:, 1:-1] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, :-2] == 0) & (voxel_array > 0))
    exposed_faces |= ((padded_world[1:-1, 1:-1, 2:] == 0) & (voxel_array > 0))
    
    return exposed_faces
