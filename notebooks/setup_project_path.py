import os
import sys

def check_and_add_project_path():
    project_path = os.environ.get('REALTIMEID_PATH')
    if project_path and project_path not in sys.path:
        sys.path.append(project_path)
        print("Project path successfully added.")
    else:
        print("Warning: 'REALTIMEID_PATH' environment variable is not set.")
