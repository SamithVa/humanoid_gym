
import argparse

args = argparse.ArgumentParser()
args.add_argument('--urdf_file', type=str, default='/data/wanshan/Desktop/robotics/humanoid-gym/resources/robots/XBot/urdf/XBot-L.urdf')

urdf_file_dir = args.parse_args().urdf_file

with open(urdf_file_dir, 'r') as urdf_file:
    urdf_content = urdf_file.read()
    def calculate_total_mass(urdf_content):
        import xml.etree.ElementTree as ET

        root = ET.fromstring(urdf_content)
        total_mass = 0.0

        for link in root.findall('link'):
            inertial = link.find('inertial')
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None and 'value' in mass_elem.attrib:
                    mass_value = float(mass_elem.attrib['value'])
                    total_mass += mass_value

        return total_mass
    total_mass = calculate_total_mass(urdf_content)
    print(f"Total mass of the robot defined in the URDF: {total_mass} kg")