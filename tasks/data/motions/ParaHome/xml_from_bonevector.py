import pickle
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

# bone_vectors: [body/lhand/rhand][njoints] ('pHipOrigin', 'jRightHip'): 0.09663602709770203
# body_global_transform.pkl (nframes, 4, 4)
# body_joint_orientations.pkl (nframes, 23, 6)
# joint_positions.pkl (nframes, 73, 3)
# joint_states.pkl [nobjects](nframes, 1) # 旋转部分用弧度表示(eg. laptop)，棱柱部分用米表示(draw)
# hand_joint_orientations.pkl (nframes, 40, 6)
# head_tips.pkl (nframes, 3)

# Load bone vectors
with open('/home/runyi/ParaHome/data/seq/s22/bone_vectors.pkl', '+rb') as f:
    bone_vectors = pickle.load(f)

# Joint information
body_vector = bone_vectors.get('body', {})
lhand_vector = bone_vectors.get('lhand', {})
rhand_vector = bone_vectors.get('rhand', {})

# Create MuJoCo XML structure
mujoco = Element('mujoco', model='humanoid')
compiler = SubElement(mujoco, 'compiler', coordinate='local')
statistic = SubElement(mujoco, 'statistic', extent='2', center='0 0 1')
option = SubElement(mujoco, 'option', timestep='0.00555')

# Worldbody
worldbody = SubElement(mujoco, 'worldbody')
light = SubElement(worldbody, 'light', cutoff='100', diffuse='1 1 1', dir='-0 0 -1.3', directional='true', exponent='1', pos='0 0 1.3', specular='.1 .1 .1')
floor = SubElement(worldbody, 'geom', conaffinity='1', condim='3', name='floor', pos='0 0 0', rgba='0.8 0.9 0.8 1', size='100 100 .2', type='plane', material='MatPlane')

# Helper function to create a body with joints and geometry
def create_body(parent, name, length, pos, geom_type='capsule'):
    body = SubElement(parent, 'body', name=name, pos=' '.join(map(str, pos)))
    geom = SubElement(body, 'geom', type=geom_type, size=str(length), pos='0 0 0')
    joint_x = SubElement(body, 'joint', name=f"{name}_x", type='hinge', pos='0 0 0', axis='1 0 0', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    joint_y = SubElement(body, 'joint', name=f"{name}_y", type='hinge', pos='0 0 0', axis='0 1 0', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    joint_z = SubElement(body, 'joint', name=f"{name}_z", type='hinge', pos='0 0 0', axis='0 0 1', stiffness='500', damping='50', armature='0.02', range='-180.0000 180.0000')
    return body

# Define which joints should be spheres
sphere_joints = {'jL5S1', 'jL4L3', 'jL1T12', 'jT9T8', 'jC1Head'}

# Build body hierarchy
bodies = {}
for (parent_name, child_name), length in {**body_vector, **lhand_vector, **rhand_vector}.items():
    # Example: Use a default position or retrieve it from your data
    pos = (0, 0, 0)  # Replace with actual position data if available

    if parent_name not in bodies:
        if parent_name == 'pHipOrigin':
            bodies[parent_name] = SubElement(worldbody, 'body', name='Pelvis', pos='0 0 0')
            SubElement(bodies[parent_name], 'freejoint', name='Pelvis')
        else:
            geom_type = 'sphere' if parent_name in sphere_joints else 'capsule'
            bodies[parent_name] = create_body(worldbody, parent_name, length, pos, geom_type)
    
    geom_type = 'sphere' if child_name in sphere_joints else 'capsule'
    bodies[child_name] = create_body(bodies[parent_name], child_name, length, pos, geom_type)

# Format XML output
xml_str = xml.dom.minidom.parseString(tostring(mujoco)).toprettyxml(indent="  ")
with open("/home/runyi/SkillMimic/skillmimic/data/assets/mjcf/mocap_parahome_s22.xml", "w") as f:
    f.write(xml_str)

print("XML file 'mocap_parahome.xml' created successfully.")
