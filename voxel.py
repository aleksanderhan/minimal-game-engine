import numpy as np
import uuid

from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import Vec3, Quat
from panda3d.bullet import BulletGenericConstraint, BulletRigidBodyNode
from panda3d.core import TransformState, NodePath
from panda3d.bullet import BulletWorld

from constants import material_properties, VoxelType, voxel_type_map
from jit import index_to_voxel_grid_coordinates, voxel_grid_coordinates_to_index
from geom import create_mesh
from world import adjust_hit_normal_to_cube, to_world_space, to_local_space 


class DynamicArbitraryVoxelObject:

    def __init__(self, 
                 voxel_array: np.ndarray, 
                 voxel_size: float, 
                 vertices: np.ndarray, 
                 indices: np.ndarray,
                 name: str = "VoxelObject",
                 debug: bool = False):
        
        #self.root = self
        #self.parent = self.root
        #self.parent_position_relative_to_grid = Vec3(0, 0, 0)

        self.voxel_array = voxel_array
        self.voxel_size = voxel_size

        self.vertices = vertices
        self.indices = indices
        
        self.debug = debug
        self.name = name
        self.id = str(uuid.uuid4())
        
        self.node_paths: dict[tuple[int, int, int], NodePath] = {}

    def get_offset(self) -> np.ndarray:
        return (np.array(self.voxel_array.shape) - 1) // 2
    
    def _set_voxel(self, ix: int, iy: int, iz: int, voxel_type: VoxelType):
        offset = self.get_offset()
        i = offset[0] + ix
        j = offset[1] + iy
        k = offset[2] + iz
        self.voxel_array[i, j, k] = voxel_type.value

    def add_voxel(self, ijk: tuple[int, int, int], hit_pos: Vec3, hit_normal: Vec3, voxel_type: VoxelType):
        hit_node_path = self.node_paths[ijk]
        orientation = hit_node_path.getQuat()

        # Convert the hit normal to the local space of the voxel
        local_hit_normal = to_local_space(hit_normal, orientation)

        # Adjust the local hit normal to align with the closest cube face
        adjusted_local_normal = adjust_hit_normal_to_cube(local_hit_normal, Quat.identQuat())

        i, j, k = ijk
        ix = int(adjusted_local_normal.x) + i
        iy = int(adjusted_local_normal.y) + j
        iz = int(adjusted_local_normal.z) + k


        print("ix, iy, iz", ix, iy, iz)



        if not (0 <= abs(ix) < self.voxel_array.shape[0] // 2 or \
                0 <= abs(iy) < self.voxel_array.shape[1] // 2 or \
                0 <= abs(iz) < self.voxel_array.shape[2] // 2):
            
            self.extend_array_uniformly()
 
        #self.voxel_array[ix, iy, iz] = voxel_type.value
        self._set_voxel(ix, iy, iz, voxel_type)


        vertices, indices = create_mesh(self.voxel_array, self.voxel_size, self.debug)


        material = material_properties[voxel_type.value]
        radius = self.voxel_size / 2
        
        shape = BulletSphereShape(radius)
        node = BulletRigidBodyNode(self.name)
        node.addShape(shape)
        node.setMass(material["mass"])
        node.setFriction(material["friction"])
        node.setRestitution(material["restitution"]) # Adjust based on testing
        node.setLinearDamping(material["linear_damping"]) # Lower values to reduce damping effect
        node.setAngularDamping(material["angular_damping"])



        #velocity = root_node_path.node().getLinearVelocity()

        return #root_node_pos, velocity, orientation


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

    def get_orientation(self) -> Quat:
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


def create_dynamic_single_voxel_object(voxel_size: int, voxel_type: VoxelType, debug: bool) -> DynamicArbitraryVoxelObject:
    voxel_array = np.zeros((1, 1, 1), np.int8)
    voxel_array[0, 0, 0] = voxel_type.value
    vertices, indices = create_mesh(voxel_array, voxel_size, debug)

    object = DynamicArbitraryVoxelObject(voxel_array, voxel_size, vertices, indices)
    node = create_dynamic_single_voxel_physics_node(object)
    return object, node           

def create_dynamic_single_voxel_physics_node(object: DynamicArbitraryVoxelObject) -> NodePath:
    voxel_type_value = object.voxel_array[0, 0, 0]
    voxel_type = voxel_type_map[voxel_type_value]
    material = material_properties[voxel_type]
    radius = object.voxel_size / 2
    
    shape = BulletSphereShape(radius)
    node = BulletRigidBodyNode(object.name)
    node.addShape(shape)
    node.setMass(material["mass"])
    node.setFriction(material["friction"])
    node.setRestitution(material["restitution"]) # Adjust based on testing
    node.setLinearDamping(material["linear_damping"]) # Lower values to reduce damping effect
    node.setAngularDamping(material["angular_damping"])

    #node_np = render.attachNewNode(node)
    #physics_world.attachRigidBody(node)
    #return node_np
    return node

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

