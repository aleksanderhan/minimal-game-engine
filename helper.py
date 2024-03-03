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



def build_robot(physicsWorld, position=(10, 10, 10)):
    # Create spherical body
    body_shape = BulletSphereShape(1)
    body_node = BulletRigidBodyNode('Body')
    body_node.addShape(body_shape)
    body_np = render.attachNewNode(body_node)
    body_np.setPos(*position)  # Position the body
    physicsWorld.attachRigidBody(body_node)

    # Add visual model for the body - using a sphere from Panda3D's basic models
    sphere_model = loader.loadModel('models/misc/sphere')
    sphere_model.reparentTo(body_np)
    sphere_model.setScale(1)  # Match the BulletSphereShape's radius

    # Function to create a visual representation of a leg segment
    def create_leg_segment_visual(parent_np, scale=(0.1, 0.1, 1), pos=Vec3(0, 0, 0)):
        card_maker = CardMaker('leg_segment')
        card_maker.setFrame(-0.5, 0.5, -0.5, 0.5)  # Create a square
        leg_segment_np = parent_np.attachNewNode(card_maker.generate())
        leg_segment_np.setScale(scale)  # Scale to act as cylinder placeholder
        leg_segment_np.setPos(pos)
        return leg_segment_np

    # Create legs with physics and visual placeholders
    leg_length = 2
    for i in range(4):  # Four legs
        angle = i * (360 / 4)
        offset = Vec3(1.5 * cos(radians(angle)), 1.5 * sin(radians(angle)), 0)

        # Upper leg part
        upper_leg_shape = BulletCylinderShape(0.1, leg_length / 2, 2)
        upper_leg_node = BulletRigidBodyNode(f'UpperLeg{i}')
        upper_leg_node.addShape(upper_leg_shape)
        upper_leg_np = render.attachNewNode(upper_leg_node)
        upper_leg_np.setPos(body_np.getPos() + offset)
        physicsWorld.attachRigidBody(upper_leg_node)

        # Add visual placeholder for the upper leg
        create_leg_segment_visual(upper_leg_np, scale=(0.1, 0.1, leg_length / 2), pos=Vec3(0, 0, -leg_length / 4))

        # Lower leg part
        lower_leg_shape = BulletCylinderShape(0.1, leg_length / 2, 2)
        lower_leg_node = BulletRigidBodyNode(f'LowerLeg{i}')
        lower_leg_node.addShape(lower_leg_shape)
        lower_leg_np = render.attachNewNode(lower_leg_node)
        lower_leg_np.setPos(upper_leg_np.getPos() + Vec3(0, 0, -leg_length / 2))
        physicsWorld.attachRigidBody(lower_leg_node)

        # Add visual placeholder for the lower leg
        create_leg_segment_visual(lower_leg_np, scale=(0.1, 0.1, leg_length / 2), pos=Vec3(0, 0, -leg_length / 4))

        # Joints 
        # Joint creation between upper and lower leg
        # Calculate the pivot point and axis for the hinge in world coordinates
        # Create the transform state for the hinge pivot and axis
        # The pivot point and axis should be specified relative to each body
        pivot_in_upper = Vec3(0, 0, -leg_length / 4)  # Adjust this pivot point as needed
        pivot_in_lower = Vec3(0, 0, leg_length / 4)  # Adjust this pivot point as needed
        axis_in_upper = Vec3(0, 1, 0)  # Adjust this axis as needed

        # Create transform states for upper and lower bodies
        transform_upper = TransformState.makePos(pivot_in_upper)
        transform_lower = TransformState.makePos(pivot_in_lower)

        # Create the hinge joint
        # Note: Panda3D's API might require the transforms to be passed in a specific way or additional adjustments
        hinge_joint = BulletHingeConstraint(upper_leg_node, lower_leg_node, transform_upper, transform_lower, True)

        # The last argument 'True' specifies use_frame_a; adjust according to your needs

        physicsWorld.attachConstraint(hinge_joint)


# Toggle generator. Returns a or b alternatingly on next()
def toggle(a, b, yield_a=True):
    while True:
        (yield a) if yield_a else (yield b)
        yield_a = not yield_a