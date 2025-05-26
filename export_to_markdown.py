#!/usr/bin/env python3
"""
Script to export Python files to a markdown document
"""

import os
import sys
from datetime import datetime

# List of files to export
files_to_export = [
    "src/webui/components/agent_settings_tab.py",
    "src/webui/components/browser_settings_tab.py", 
    "src/webui/components/browser_use_agent_tab.py",
    "src/webui/interface.py",
    "src/webui/components/deep_research_agent_tab.py",
    "src/webui/components/load_save_config_tab.py",
    "webui.py"  # Including the main webui.py file as well
]

def export_to_markdown(output_file="exported_code.md"):
    """Export all specified Python files to a markdown document"""
    
    with open(output_file, 'w', encoding='utf-8') as md_file:
        # Write header
        md_file.write("# Browser Use WebUI - Exported Python Files\n\n")
        md_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md_file.write("## Table of Contents\n\n")
        
        # Write table of contents
        for i, file_path in enumerate(files_to_export, 1):
            file_name = os.path.basename(file_path)
            md_file.write(f"{i}. [{file_name}](#{file_name.replace('.', '').replace('_', '-')})\n")
        
        md_file.write("\n---\n\n")
        
        # Export each file
        for file_path in files_to_export:
            file_name = os.path.basename(file_path)
            print(f"Exporting {file_path}...")
            
            md_file.write(f"## {file_name}\n\n")
            md_file.write(f"**Path:** `{file_path}`\n\n")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as py_file:
                        content = py_file.read()
                        line_count = len(content.splitlines())
                        
                    md_file.write(f"**Lines:** {line_count}\n\n")
                    md_file.write("```python\n")
                    md_file.write(content)
                    md_file.write("\n```\n\n")
                    md_file.write("---\n\n")
                    
                except Exception as e:
                    md_file.write(f"**Error reading file:** {str(e)}\n\n")
                    md_file.write("---\n\n")
            else:
                md_file.write("**Error:** File not found\n\n")
                md_file.write("---\n\n")
    
    print(f"\nExport completed! Output saved to: {output_file}")
    return output_file

def main():
    """Main function to run the export"""
    output_file = "browser_use_webui_export.md"
    
    # Check if we're in the right directory
    if not os.path.exists("src/webui/interface.py"):
        print("Error: Please run this script from the Browser-Use-retro-web-ui directory")
        sys.exit(1)
    
    # Export to markdown
    exported_file = export_to_markdown(output_file)
    
    # Ask if user wants to open the file
    response = input(f"\nDo you want to open {exported_file}? (y/n): ").lower()
    if response == 'y':
        # Try to open the file with the default system editor
        if sys.platform == "win32":
            os.startfile(exported_file)
        elif sys.platform == "darwin":  # macOS
            os.system(f"open {exported_file}")
        else:  # linux variants
            os.system(f"xdg-open {exported_file}")

if __name__ == "__main__":
    main() 