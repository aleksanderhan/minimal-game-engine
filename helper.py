import numpy as np
import noise
import pyautogui
import argparse

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.showbase.InputStateGlobal import inputState
from direct.gui.OnscreenText import OnscreenText
from direct.actor.Actor import Actor

from panda3d.core import (
    CardMaker, Vec3
)
from panda3d.bullet import (
    BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletSphereShape,
    BulletTriangleMesh, BulletTriangleMeshShape, BulletHeightfieldShape  # Conceptual
)
from panda3d.bullet import BulletRigidBodyNode, BulletCapsuleShape
from panda3d.core import NodePath
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletSphereShape, BulletCylinderShape, BulletHingeConstraint, BulletDebugNode
from panda3d.core import Vec3, TransformState
from math import cos, sin, radians
from panda3d.core import Texture
import random
from panda3d.core import TransformState, Vec3
from panda3d.core import Vec3, TransformState
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletSphereShape, BulletCylinderShape, BulletHingeConstraint
from direct.showbase.ShowBase import ShowBase
from math import cos, sin, radians
from panda3d.core import NodePath, CollisionNode, CollisionSphere, CollisionCapsule
from panda3d.bullet import BulletRigidBodyNode, BulletSphereShape, BulletCapsuleShape, BulletWorld
from panda3d.core import TransformState, Vec3
from panda3d.core import Vec3, TransformState
from panda3d.bullet import BulletConeTwistConstraint, BulletHingeConstraint

from panda3d.core import NodePath, Vec3
from panda3d.bullet import BulletWorld, BulletRigidBodyNode, BulletSphereShape, BulletCapsuleShape, BulletConeTwistConstraint, BulletHingeConstraint
from direct.showbase.ShowBase import ShowBase

class Robot:
    def __init__(self, render, physics_world, position, radius=0.5):
        self.shpere_radius = radius
        self.render = render
        self.physics_world = physics_world
        self.position = position
        self.body = None
        self.legs = []
        self.joints = []
        self.create_robot_body()
        self.create_robot_legs()

    def create_robot_body(self):
        body_shape = BulletSphereShape(0.4)  # Radius of the body
        self.body_node = BulletRigidBodyNode('robot_body')
        self.body_node.setMass(1.0)  # Set a non-zero mass
        self.body_node.addShape(body_shape)
        self.body = self.render.attachNewNode(self.body_node)
        self.body.setPos(self.position)  # Starting position of the body
        self.physics_world.attachRigidBody(self.body_node)

    def create_robot_legs(self):
        leg_positions = [(self.shpere_radius, self.shpere_radius, 0), (-self.shpere_radius, self.shpere_radius, 0), (self.shpere_radius, -self.shpere_radius, 0), (-self.shpere_radius, -self.shpere_radius, 0)]
        for i, (x, y, z) in enumerate(leg_positions):
            # Create upper leg
            upper_leg_shape = BulletCapsuleShape(0.05, 0.5)  # Radius and cylinder height
            upper_leg_node = BulletRigidBodyNode(f'upper_leg_{i}')
            upper_leg_node.setMass(0.1)
            upper_leg_node.addShape(upper_leg_shape)
            upper_leg = self.render.attachNewNode(upper_leg_node)
            upper_leg.setPos(self.body.getX() + x, self.body.getY() + y, self.body.getZ() + z - self.shpere_radius)
            self.physics_world.attachRigidBody(upper_leg_node)

            # Create lower leg
            lower_leg_shape = BulletCapsuleShape(0.05, 0.5)  # Adjust dimensions as needed
            lower_leg_node = BulletRigidBodyNode(f'lower_leg_{i}')
            lower_leg_node.setMass(0.1)
            lower_leg_node.addShape(lower_leg_shape)
            lower_leg = self.render.attachNewNode(lower_leg_node)
            lower_leg.setPos(upper_leg.getX(), upper_leg.getY(), upper_leg.getZ() - self.shpere_radius)
            self.physics_world.attachRigidBody(lower_leg_node)

            # Create joints
            # Connect upper leg to the body
            pivot_in_body = Vec3(x, y, z - self.shpere_radius)
            pivot_in_upper_leg = Vec3(0, 0, 0.25)
            # Assuming pivot_in_body and pivot_in_upper_leg are Vec3 objects for the pivot positions
            transform_in_body = TransformState.makePosHpr(Vec3(0, 0, 0), pivot_in_body)
            transform_in_upper_leg = TransformState.makePosHpr(Vec3(0, 0, 0), pivot_in_upper_leg)
            upper_leg_joint = BulletConeTwistConstraint(self.body_node, upper_leg_node, transform_in_body, transform_in_upper_leg)
            cone_twist_joint = BulletConeTwistConstraint(self.body_node, upper_leg_node, transform_in_body, transform_in_upper_leg)
            cone_twist_joint.setLimit(0, 0, 0)  # Example values, adjust as needed
            self.physics_world.attachConstraint(cone_twist_joint)

            # Connect lower leg to upper leg
            pivot_in_upper = Vec3(0, 0, -0.25)
            pivot_in_lower = Vec3(0, 0, 0.25)
            hinge_joint = BulletHingeConstraint(upper_leg_node, lower_leg_node, pivot_in_upper, pivot_in_lower, Vec3(1, 0, 0), Vec3(1, 0, 0))
            hinge_joint.setLimit(0, 0)  # Example values, adjust as needed
            self.physics_world.attachConstraint(hinge_joint)

            self.legs.append((upper_leg, lower_leg))
            self.joints.extend([upper_leg_joint, hinge_joint])

    def set_position(self, x, y, z):
        self.body.node().setKinematic(True)
        self.body.setPos(x, y, z)
        self.body.node().setKinematic(False)


# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a