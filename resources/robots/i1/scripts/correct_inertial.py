import re

def extract_inertial_properties(raw_string):
    """
    Extract inertial properties from raw mass properties text and return XML.
    Handles missing fields gracefully with None defaults.
    
    Returns:
        str: XML string with inertial properties, or None if extraction fails
    """
    try:
        # Use regex to robustly extract values with optional whitespace variations
        mass_match = re.search(r'Mass\s*=\s*([\d.eE+-]+)', raw_string)
        x_match = re.search(r'X\s*=\s*([\d.eE+-]+)', raw_string)
        y_match = re.search(r'Y\s*=\s*([\d.eE+-]+)', raw_string)
        z_match = re.search(r'Z\s*=\s*([\d.eE+-]+)', raw_string)
        lxx_match = re.search(r'Lxx\s*=\s*([\d.eE+-]+)', raw_string)
        lxy_match = re.search(r'Lxy\s*=\s*([\d.eE+-]+)', raw_string)
        lxz_match = re.search(r'Lxz\s*=\s*([\d.eE+-]+)', raw_string)
        lyy_match = re.search(r'Lyy\s*=\s*([\d.eE+-]+)', raw_string)
        lyz_match = re.search(r'Lyz\s*=\s*([\d.eE+-]+)', raw_string)
        lzz_match = re.search(r'Lzz\s*=\s*([\d.eE+-]+)', raw_string)
        
        # Extract values or use 0 as default
        mass = str(float(mass_match.group(1))) if mass_match else "0"
        cog_x = str(float(x_match.group(1))) if x_match else "0"
        cog_y = str(float(y_match.group(1))) if y_match else "0"
        cog_z = str(float(z_match.group(1))) if z_match else "0"
        ixx = str(float(lxx_match.group(1))) if lxx_match else "0"
        ixy = str(-float(lxy_match.group(1))) if lxy_match else "0"
        ixz = str(-float(lxz_match.group(1))) if lxz_match else "0"
        iyy = str(float(lyy_match.group(1))) if lyy_match else "0"
        iyz = str(-float(lyz_match.group(1))) if lyz_match else "0"
        izz = str(float(lzz_match.group(1))) if lzz_match else "0"

        # Generate XML string
        output_string = f"""    <inertial>
        <origin
            xyz="{cog_x} {cog_y} {cog_z}"
            rpy="0 0 0" />
        <mass
            value="{mass}" />
        <inertia
            ixx="{ixx}"
            ixy="{ixy}"
            ixz="{ixz}"
            iyy="{iyy}"
            iyz="{iyz}"
            izz="{izz}" />
    </inertial>"""

        return output_string
    except Exception as e:
        print(f"Error extracting inertial properties: {e}")
        return None

def parse_all_joints(txt_file_path):
    """
    Parse all joints from text file and extract their inertial properties.
    
    Args:
        txt_file_path (str): Path to the inertia text file
        
    Returns:
        dict: Dictionary with joint names as keys and XML strings as values
    """
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by joint names (pattern: word ending with _joint followed by newline)
    joint_blocks = re.split(r'^(?=\S+_joint\s*\n)', content, flags=re.MULTILINE)
    
    results = {}
    
    for block in joint_blocks:
        if not block.strip():
            continue
            
        # Extract joint name (first line)
        lines = block.strip().split('\n')
        joint_name = lines[0].strip()
        
        # Skip if not a valid joint name
        if not joint_name.endswith('_joint'):
            continue
        
        # Get the rest of the block (inertial data)
        inertial_data = '\n'.join(lines[1:])
        
        # Extract inertial properties
        xml = extract_inertial_properties(inertial_data)
        
        if xml:
            results[joint_name] = xml
            print(f"✓ Extracted {joint_name}")
        else:
            print(f"✗ Failed to extract {joint_name}")
    
    return results


def output_results(results, output_file_path):
    """
    Write all inertial properties to an output XML file.
    
    Args:
        results (dict): Dictionary of joint names and XML strings
        output_file_path (str): Path to output file
    """
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write('<?xml version="1.0"?>\n')
        file.write('<!-- Inertial properties extracted from CAD models -->\n')
        file.write(f'<!-- Total joints: {len(results)} -->\n\n')
        
        for joint_name, xml in sorted(results.items()):
            file.write(f'<!-- {joint_name} -->\n')
            file.write(xml)
            file.write('\n\n')
    
    print(f"\n✓ Output written to {output_file_path}")


if __name__ == '__main__':
    txt_file_path = "./inertia_1028.txt"
    output_file_path = "./inertials_output.xml"
    
    print("Parsing all joints from inertia_1028.txt...\n")
    results = parse_all_joints(txt_file_path)
    
    print(f"\n{'='*50}")
    print(f"Total joints extracted: {len(results)}")
    print(f"{'='*50}\n")
    
    # Print each joint's XML
    for joint_name, xml in sorted(results.items()):
        print(f"{joint_name}:")
        print(xml)
        print()
    
    # Write to output file
    output_results(results, output_file_path)