import numpy as np
import uuid

from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape
from panda3d.core import Vec3, Quat, Point3
from panda3d.bullet import BulletGenericConstraint, BulletRigidBodyNode
from panda3d.core import TransformState, NodePath
from panda3d.bullet import BulletWorld

from constants import material_properties, VoxelType, voxel_type_map
from geom import create_mesh
from world import adjust_spherical_normal_to_cube, to_world_space, to_local_space, project_sphere_point_to_cube

from geom import create_geometry
from panda3d.bullet import (
    BulletBoxShape,
    BulletDebugNode,
    BulletRigidBodyNode,
    BulletSphereShape,
    BulletTriangleMesh,
    BulletTriangleMeshShape,
    BulletWorld,
    BulletShape,
)
from panda3d.core import (
    NodePath,
    Vec3,
    Quat,
    Point3,
)


color_map = {
    "red": (1, 0, 0, 1),
    "green": (0, 1,0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
}


class DynamicObject:
    def __init__(
        self,
        scale: float,
        shape: BulletShape,
        node: BulletRigidBodyNode,
        model_file: str,
        name: str = "object",
        kind: str = "object",
        color_name: str = "unknown",
        tags: set[str] | None = None,
        debug: bool = False,
    ):
        self.scale = scale
        self.shape = shape
        self.node = node
        self.model_file = model_file

        self.node_path: NodePath | None = None

        self.name = name
        self.kind = kind
        self.color_name = color_name
        self.tags = tags or set()

        self.debug = debug
        self.id = str(uuid.uuid4())
        

    def __str__(self) -> str:
        return "DynamicObject: " + self.name + " id: " + self.id + "\n" \
            + " position:" + str(self.get_position()) + " orientation:" + str(self.get_orientation()) + " velocity:" + str(self.get_velocity()) + "\n" \
            + " vertices:" + str(len(self.vertices)) + " indices:" + str(len(self.indices)) + "\n" \

    def _build_node(self) -> BulletRigidBodyNode:
        node = BulletRigidBodyNode()
        node.setMass(1.0)
        return node
    
    @staticmethod
    def create_sphere(
        scale: float,
        mass: float,
        name: str = "ball",
        color_name: str = "unknown",
        tags: set[str] | None = None,
    ):
        sphere_shape = BulletSphereShape(scale)
        sphere_node = BulletRigidBodyNode(name)
        sphere_node.addShape(sphere_shape)
        sphere_node.setMass(mass)

        return DynamicObject(
            scale=scale,
            shape=sphere_shape,
            node=sphere_node,
            model_file="models/misc/sphere.egg",
            name=name,
            kind="ball",
            color_name=color_name,
            tags=tags or {"physics_object", "movable"},
        )

    def get_position(self) -> Point3:
        return self.node_path.getPos()

    def get_orientation(self) -> Quat:
        return self.node_path.getQuat()
    
    def get_velocity(self) -> Vec3:
        return self.node_path.node().getLinearVelocity()
    
    def set_velocity(self, velocity: Vec3):
        node = self.node_path.node()
        node.setLinearVelocity(velocity)

    def enable_ccd(self):
        node = self.node_path.node()
        #voxel_diagonal = math.sqrt(3 * self.voxel_size**2)
        ccd_radius = self.voxel_size / 2 #voxel_diagonal / 2
        node.setCcdMotionThreshold(1e-7)
        node.setCcdSweptSphereRadius(ccd_radius)


class DynamicArbitraryVoxelObject:

    def __init__(self, 
                 voxel_array: np.ndarray, 
                 voxel_size: float,
                 name: str = "DynamicArbitraryVoxelObject",
                 debug: bool = False):

        self.voxel_array = voxel_array
        self.voxel_size = voxel_size

        self.vertices, self.indices = create_mesh(voxel_array, voxel_size, debug)
        
        self.debug = debug
        self.name = name
        self.id = str(uuid.uuid4())
        
        self.node: BulletRigidBodyNode = self._build_node()
        self.node_np: NodePath = None

    def __str__(self) -> str:
        return "DynamicArbitraryVoxelObject: " + self.name + " id: " + self.id + "\n" \
            + " position:" + str(self.get_position()) + " orientation:" + str(self.get_orientation()) + " velocity:" + str(self.get_velocity()) + "\n" \
            + " vertices:" + str(len(self.vertices)) + " indices:" + str(len(self.indices)) + "\n" \
            + " voxel_array:" + str(self.voxel_array.shape)

    def _get_offset(self) -> np.ndarray:
        return (np.array(self.voxel_array.shape) - 1) // 2
    
    def _set_voxel(self, ix: int, iy: int, iz: int, voxel_type: VoxelType):
        offset = self._get_offset()
        i = offset[0] + ix
        j = offset[1] + iy
        k = offset[2] + iz
        self.voxel_array[i, j, k] = voxel_type.value

    def add_voxel(self, hit_pos: Vec3, hit_normal: Vec3, voxel_type: VoxelType):
        
        print("node.pos", self.node_np.getPos())
        print("hit_pos", hit_pos)

        relative_pos = hit_pos - self.node_np.getPos()
        print("relative_pos", relative_pos)

        adjusted_pos = project_sphere_point_to_cube(relative_pos)
        print("adjusted_pos", adjusted_pos)

        orientation = self.node_np.getQuat()

        # Convert the hit normal to the local space of the voxel
        local_hit_normal = to_local_space(hit_normal, orientation)

        # Adjust the local hit normal to align with the closest cube face
        adjusted_local_normal = adjust_spherical_normal_to_cube(local_hit_normal, Quat.identQuat())
        ix = int(adjusted_local_normal.x)
        iy = int(adjusted_local_normal.y)
        iz = int(adjusted_local_normal.z)


        print("ix, iy, iz", ix, iy, iz)

        if not (0 <= abs(ix) < self.voxel_array.shape[0] // 2 or \
                0 <= abs(iy) < self.voxel_array.shape[1] // 2 or \
                0 <= abs(iz) < self.voxel_array.shape[2] // 2):
            
            self._extend_array_uniformly()
 
        #self.voxel_array[ix, iy, iz] = voxel_type.value
        self._set_voxel(ix, iy, iz, voxel_type)

        self.vertices, self.indices = create_mesh(self.voxel_array, self.voxel_size, self.debug)        
        self.node = self._build_node()


    def _build_node(self) -> BulletRigidBodyNode:
        node = BulletRigidBodyNode()
        node.setMass(1.0)

        polulated_indices = np.argwhere(self.voxel_array)
        radius = self.voxel_size / 2
        offset = self._get_offset()

        for i, j, k in polulated_indices:
            ix = i - offset[0]
            iy = j - offset[1]
            iz = k - offset[2]
            shape = BulletSphereShape(radius)
            node.addShape(shape, TransformState.makePos(Point3(ix, iy, iz)))

        return node

    def _extend_array_uniformly(self):
        # Specify the padding width of 1 for all sides of all dimensions
        pad_width = [(1, 1)] * 3  # Padding for depth, rows, and columns
        
        # Pad the array with 0's on all sides
        self.voxel_array = np.pad(self.voxel_array, pad_width=pad_width, mode='constant', constant_values=0)

    def get_position(self) -> Point3:
        return self.node_np.getPos()

    def get_orientation(self) -> Quat:
        return self.node_np.getQuat()
    
    def get_velocity(self) -> Vec3:
        return self.node_np.node().getLinearVelocity()
    
    def set_velocity(self, velocity: Vec3):
        node = self.node_np.node()
        node.setLinearVelocity(velocity)

    def enable_ccd(self):
        node = self.node_np.node()
        #voxel_diagonal = math.sqrt(3 * self.voxel_size**2)
        ccd_radius = self.voxel_size / 2 #voxel_diagonal / 2
        node.setCcdMotionThreshold(1e-7)
        node.setCcdSweptSphereRadius(ccd_radius)

    @staticmethod
    def create_dynamic_single_voxel_object(voxel_size: int, voxel_type: VoxelType, debug: bool) -> "DynamicArbitraryVoxelObject":
        voxel_array = np.zeros((1, 1, 1), np.int8)
        voxel_array[0, 0, 0] = voxel_type.value

        return DynamicArbitraryVoxelObject(voxel_array, voxel_size)     


class ObjectManager:

    def __init__(self, game_engine):
        self.game_engine = game_engine
        self.object_map = {}

    def register_object(self, 
                        dynamic_object: DynamicObject | DynamicArbitraryVoxelObject,
                        position: Vec3, 
                        velocity = Vec3(0, 0, 0), 
                        orientation = Quat(0, 0, 0, 0),
                        ccd=False):
        
        node = dynamic_object.node
        node_path = self.game_engine.render.attachNewNode(node)
        self.game_engine.physics_world.attachRigidBody(node)
        node_path.setPos(position)
        node_path.setQuat(orientation)

        if isinstance(dynamic_object, DynamicArbitraryVoxelObject):
            geom_node_path = create_geometry(object.vertices, object.indices)
            geom_node_path.reparentTo(self.game_engine.render)
            geom_node_path.reparentTo(node_path)

        # Load the sphere model and attach it to the physics node
        sphere = self.game_engine.loader.loadModel(dynamic_object.model_file)
        sphere.setScale(dynamic_object.scale)  # Adjust the scale as needed

        color = color_map.get(dynamic_object.color_name, (1, 1, 1, 1))
        sphere.setColor(*color)  # Set the sphere's color
        sphere.reparentTo(node_path)  # Correctly attach the model to the NodePath

        node_path.setPythonTag("object", dynamic_object)
        dynamic_object.node_path = node_path
        self.object_map[dynamic_object.id] = dynamic_object

        #geom_np = create_geometry(dynamic_object.vertices, dynamic_object.indices)
        #geom_np.reparentTo(self.game_engine.render)
        #geom_np.reparentTo(node_np)

        dynamic_object.set_velocity(velocity)
        ccd = velocity.length() > 50
        if ccd:
            dynamic_object.enable_ccd()


    def deregister_object(self, dynamic_object):
        self.game_engine.physics_world.removeRigidBody(dynamic_object.node_path.node())
        dynamic_object.node_path.removeNode()
        del self.object_map[dynamic_object.id]

    def update_object(self, dynamic_object, position, velocity, orientation):
        self.deregister_object(dynamic_object)
        self.register_object(dynamic_object, position, velocity, orientation)

    def get_object(self, object_id: str) -> DynamicObject | None:
        return self.object_map.get(object_id)


    def get_all_objects(self) -> list[DynamicObject]:
        return list(self.object_map.values())


    def get_objects_except(self, excluded_object_id: str) -> list[DynamicObject]:
        return [
            dynamic_object
            for dynamic_object in self.object_map.values()
            if dynamic_object.id != excluded_object_id
    ]

