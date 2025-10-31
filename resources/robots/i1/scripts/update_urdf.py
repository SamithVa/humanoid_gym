import xml.etree.ElementTree as ET
import re

corrected_inertia_urdf_path = "./inertials_output.xml"
original_urdf_path = "./i1_1028.urdf"
output_urdf_path = "./i1_1028_updated.urdf"


def parse_inertials_xml(xml_file_path):
    """
    Parse the corrected inertials XML file and extract inertial properties by link name.
    
    Args:
        xml_file_path (str): Path to the inertials_output.xml file
        
    Returns:
        dict: Dictionary with link names as keys and inertial XML strings as values
    """
    inertials_dict = {}
    
    with open(xml_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by comment pattern to identify each link's inertial block
    # Pattern: <!-- link_name --> followed by inertial XML
    pattern = r'<!-- (\w+) -->\s*(<inertial>.*?</inertial>)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    for match in matches:
        link_name = match.group(1)
        inertial_xml = match.group(2)
        inertials_dict[link_name] = inertial_xml
        print(f"✓ Loaded inertial for: {link_name}")
    
    print(f"\nTotal inertials loaded: {len(inertials_dict)}\n")
    return inertials_dict


def replace_inertials_in_urdf(urdf_file_path, inertials_dict, output_file_path):
    """
    Replace all inertial elements in the URDF file with corrected values.
    
    Args:
        urdf_file_path (str): Path to the original URDF file
        inertials_dict (dict): Dictionary of corrected inertials by link name
        output_file_path (str): Path to save the updated URDF
    """
    with open(urdf_file_path, 'r', encoding='utf-8') as file:
        urdf_content = file.read()
    
    # Parse the URDF to find all links and their inertial sections
    root = ET.fromstring(urdf_content)
    
    # Track replaced inertials
    replaced_count = 0
    not_found = []
    
    # Iterate through all links in the URDF
    for link in root.findall('.//link'):
        link_name = link.get('name')
        
        # Check if we have corrected inertial data for this link
        if link_name in inertials_dict:
            # Find and remove the existing inertial element
            existing_inertial = link.find('inertial')
            if existing_inertial is not None:
                link.remove(existing_inertial)
            
            # Parse the new inertial XML and add it
            new_inertial_xml = inertials_dict[link_name]
            new_inertial_element = ET.fromstring(new_inertial_xml)
            link.insert(0, new_inertial_element)  # Insert at beginning
            
            replaced_count += 1
            print(f"✓ Replaced inertial for link: {link_name}")
        else:
            # Keep the original inertial if not in corrected list
            if link.find('inertial') is not None:
                not_found.append(link_name)
    
    # Write the updated URDF to file
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(output_file_path, encoding='utf-8', xml_declaration=True)
    
    print(f"\n{'='*60}")
    print(f"Total links updated: {replaced_count}")
    if not_found:
        print(f"Links kept with original inertials: {len(not_found)}")
        for link in not_found:
            print(f"  - {link}")
    print(f"{'='*60}")
    print(f"\n✓ Updated URDF saved to: {output_file_path}")


if __name__ == '__main__':
    print("="*60)
    print("URDF Inertial Properties Updater")
    print("="*60 + "\n")
    
    # Parse corrected inertials
    print("Loading corrected inertials from XML...\n")
    inertials_dict = parse_inertials_xml(corrected_inertia_urdf_path)
    
    # Replace inertials in URDF
    print("Updating URDF with corrected inertials...\n")
    replace_inertials_in_urdf(original_urdf_path, inertials_dict, output_urdf_path)



