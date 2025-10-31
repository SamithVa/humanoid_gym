# calculate all mass in urdf file
import re
import xml.etree.ElementTree as ET

urdf_file_path = "./i1_1028_updated.urdf"


def extract_mass_from_urdf(urdf_file_path):
    """
    Extract all mass values from links in the URDF file and calculate the total.
    
    Args:
        urdf_file_path (str): Path to the URDF file
        
    Returns:
        tuple: (total_mass, mass_dict) where mass_dict has link names as keys and mass values as values
    """
    try:
        # Parse the URDF XML file
        tree = ET.parse(urdf_file_path)
        root = tree.getroot()
        
        mass_dict = {}
        total_mass = 0.0
        
        # Find all links and extract their mass values
        for link in root.findall('.//link'):
            link_name = link.get('name')
            inertial = link.find('inertial')
            
            if inertial is not None:
                mass_elem = inertial.find('mass')
                if mass_elem is not None:
                    mass_value_str = mass_elem.get('value')
                    if mass_value_str:
                        try:
                            mass_value = float(mass_value_str)
                            mass_dict[link_name] = mass_value
                            total_mass += mass_value
                        except ValueError:
                            print(f"Warning: Could not convert mass value for {link_name}: {mass_value_str}")
            else:
                print(f"Note: {link_name} has no inertial properties")
        
        return total_mass, mass_dict
    
    except Exception as e:
        print(f"Error parsing URDF file: {e}")
        return 0.0, {}


def display_mass_breakdown(mass_dict, total_mass):
    """
    Display a detailed breakdown of all link masses.
    
    Args:
        mass_dict (dict): Dictionary of link names and their masses
        total_mass (float): Total mass of all links
    """
    print("="*70)
    print("URDF MASS BREAKDOWN")
    print("="*70 + "\n")
    
    print(f"{'Link Name':<30} {'Mass (kg)':<20} {'Percentage':<15}")
    print("-"*70)
    
    # Sort by mass descending for better visualization
    sorted_links = sorted(mass_dict.items(), key=lambda x: x[1], reverse=True)
    
    for link_name, mass_value in sorted_links:
        percentage = (mass_value / total_mass * 100) if total_mass > 0 else 0
        print(f"{link_name:<30} {mass_value:<20.6f} {percentage:<14.2f}%")
    
    print("-"*70)
    print(f"{'TOTAL MASS':<30} {total_mass:<20.6f} {100.0:<14.2f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    print("Calculating total mass from URDF file...\n")
    
    total_mass, mass_dict = extract_mass_from_urdf(urdf_file_path)
    
    if mass_dict:
        print(f"✓ Successfully extracted mass from {len(mass_dict)} links\n")
        display_mass_breakdown(mass_dict, total_mass)
        
        print(f"📊 VERIFICATION SUMMARY:")
        print(f"   Number of links with mass: {len(mass_dict)}")
        print(f"   Total mass of robot: {total_mass:.6f} kg")
        print(f"   Average mass per link: {total_mass/len(mass_dict):.6f} kg")
        print(f"   Heaviest component: {max(mass_dict.items(), key=lambda x: x[1])[0]}")
        print(f"   Lightest component: {min(mass_dict.items(), key=lambda x: x[1])[0]}\n")
    else:
        print("✗ No mass data found in URDF file")

