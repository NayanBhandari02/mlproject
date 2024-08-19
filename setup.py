from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]
    
    print(f"Requirements read from file: {requirements}")  # Debug print
    
    if '-e .' in requirements:
        requirements.remove('-e .')
    
    return requirements 

setup(
name='mlproject',
version='0.0.1',
author='Nayan Vipin Bhandari',
author_email='bhandarinayan02@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)