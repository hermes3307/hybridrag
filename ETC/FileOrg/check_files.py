import os
import re
from collections import Counter
import datetime
import json
import glob

def analyze_directory_structure(root_path):
    """
    Recursively analyze directories to identify file patterns and outliers,
    then provide recommendations.
    
    Args:
        root_path (str): The root directory to start analysis from
        
    Returns:
        dict: Analysis results and recommendations
    """
    print(f"Starting analysis of directory: {root_path}")
    print("-" * 80)
    
    # Store analysis results and recommendations
    results = {
        "delete_recommendations": [],
        "rename_recommendations": [],
        "directory_analyses": []
    }
    
    # Walk through directory tree recursively
    for dirpath, dirnames, filenames in os.walk(root_path):
        if not filenames:
            continue  # Skip empty directories
        
        # Get relative path for cleaner output
        rel_path = os.path.relpath(dirpath, root_path)
        if rel_path == '.':
            rel_path = 'Root'
            
        print(f"\nDirectory: {rel_path}")
        print(f"Contains {len(filenames)} files")
        
        dir_analysis = {
            "path": rel_path,
            "file_count": len(filenames),
            "outliers": [],
            "dominant_pattern": None,
            "recommendations": []
        }
        
        # Extract base filenames without extensions and count occurrences
        base_names = [os.path.splitext(f)[0] for f in filenames]
        base_counter = Counter(base_names)
        
        # Find the dominant filename pattern (if any)
        if base_counter:
            most_common_base, count = base_counter.most_common(1)[0]
            percentage = (count / len(filenames)) * 100
            
            if percentage >= 50:  # If more than 50% have the same base name
                print(f"Dominant file pattern: {most_common_base}.*  ({count} files, {percentage:.1f}%)")
                dir_analysis["dominant_pattern"] = f"{most_common_base}.*"
                
                # Find files with different base names (outliers)
                outliers = [f for f in filenames if os.path.splitext(f)[0] != most_common_base]
                if outliers:
                    print(f"Files with different patterns ({len(outliers)}):")
                    for outlier in outliers:
                        print(f"  - {outlier}")
                        dir_analysis["outliers"].append(outlier)
                        
                        # Generate recommendations
                        full_path = os.path.join(dirpath, outlier)
                        outlier_base, outlier_ext = os.path.splitext(outlier)
                        
                        # Check if the extension matches common files in the directory
                        common_extensions = [os.path.splitext(f)[1] for f in filenames 
                                         if os.path.splitext(f)[0] == most_common_base]
                        
                        if common_extensions and outlier_ext in common_extensions:
                            # If extension matches but name doesn't, recommend rename
                            new_name = f"{most_common_base}{outlier_ext}"
                            rename_rec = {
                                "path": os.path.join(rel_path, outlier),
                                "current_name": outlier,
                                "suggested_name": new_name,
                                "full_path": full_path,
                                "new_full_path": os.path.join(dirpath, new_name)
                            }
                            results["rename_recommendations"].append(rename_rec)
                            dir_analysis["recommendations"].append(
                                f"Rename '{outlier}' to '{new_name}' to match pattern"
                            )
                        else:
                            # If both name and extension don't match, consider deletion
                            delete_rec = {
                                "path": os.path.join(rel_path, outlier),
                                "filename": outlier,
                                "full_path": full_path,
                                "reason": "Doesn't match dominant pattern"
                            }
                            results["delete_recommendations"].append(delete_rec)
                            dir_analysis["recommendations"].append(
                                f"Consider deleting '{outlier}' as it doesn't match the pattern"
                            )
            else:
                # No dominant pattern
                print("No dominant file pattern found. File types:")
                # Group by extension
                extensions = [os.path.splitext(f)[1].lower() for f in filenames if os.path.splitext(f)[1]]
                ext_counter = Counter(extensions)
                
                dir_analysis["file_types"] = []
                for ext, count in ext_counter.most_common():
                    print(f"  {ext}: {count} files")
                    dir_analysis["file_types"].append({"extension": ext, "count": count})
                
                # If there are too many different types, suggest organizing
                if len(ext_counter) > 3:
                    dir_analysis["recommendations"].append(
                        f"Consider organizing files by moving different file types to separate subdirectories"
                    )
        
        results["directory_analyses"].append(dir_analysis)
    
    return results

def generate_todo_list(results):
    """Generate a formatted to-do list based on analysis results"""
    print("\n" + "=" * 80)
    print("DECISION TO-DO LIST")
    print("=" * 80)
    
    # Deletion recommendations
    if results["delete_recommendations"]:
        print("\n1. Files Recommended for Deletion:")
        for i, rec in enumerate(results["delete_recommendations"], 1):
            print(f"   {i}. Delete: {rec['path']}")
            print(f"      Reason: {rec['reason']}")
    else:
        print("\n1. No files recommended for deletion")
    
    # Rename recommendations
    if results["rename_recommendations"]:
        print("\n2. Files Recommended for Renaming:")
        for i, rec in enumerate(results["rename_recommendations"], 1):
            print(f"   {i}. Rename: {rec['path']}")
            print(f"      From: {rec['current_name']}")
            print(f"      To:   {rec['suggested_name']}")
    else:
        print("\n2. No files recommended for renaming")
    
    # Directory-specific recommendations
    print("\n3. Directory-Specific Recommendations:")
    
    dirs_with_recommendations = [dir_analysis for dir_analysis in results["directory_analyses"] 
                               if dir_analysis["recommendations"]]
    
    if dirs_with_recommendations:
        for i, dir_analysis in enumerate(dirs_with_recommendations, 1):
            print(f"   {i}. Directory: {dir_analysis['path']}")
            for j, rec in enumerate(dir_analysis["recommendations"], 1):
                print(f"      {j}. {rec}")
    else:
        print("   No directory-specific recommendations")
    
    # Summary
    total_delete = len(results["delete_recommendations"])
    total_rename = len(results["rename_recommendations"])
    
    print("\n" + "-" * 80)
    print(f"Summary: {total_delete} files recommended for deletion, {total_rename} files recommended for renaming")
    print("-" * 80)

def find_backup_files(root_path):
    """Find all backup files created during operations"""
    backup_pattern = "*.backup_*"
    backup_files = []
    
    for dirpath, _, _ in os.walk(root_path):
        for backup_file in glob.glob(os.path.join(dirpath, backup_pattern)):
            backup_files.append(backup_file)
    
    return backup_files

def generate_action_script(results, root_path):
    """Generate a Python script to execute the recommendations with proper encoding"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    script_filename = f"execute_recommendations_{timestamp}.py"
    
    script_content = """# -*- coding: utf-8 -*-
import os
import shutil
import datetime
import json
import sys
import glob
import time

# Ensure proper encoding for console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # For Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# This will store all backup files created during the operation
BACKUP_FILES = []

def log_action(log_file, action, path, details=""):
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {action}: {path} {details}\\n")

def find_backup_files(root_path):
    \"\"\"Find all backup files created during operations\"\"\"
    backup_pattern = "*.backup_*"
    backup_files = []
    
    for dirpath, _, _ in os.walk(root_path):
        for backup_file in glob.glob(os.path.join(dirpath, backup_pattern)):
            backup_files.append(backup_file)
    
    return backup_files
    
def main():
    # Create log file
    log_file = f"file_actions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"File Actions Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write("-" * 80 + "\\n")
    
    print("Starting execution of recommended actions...")
    print(f"Actions will be logged to {log_file}")
    
    # Store the root path for backup file search
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
"""
    
    # Write JSON files for paths instead of embedding them in the script
    if results["delete_recommendations"]:
        delete_paths = []
        delete_info = []
        
        for i, rec in enumerate(results["delete_recommendations"]):
            delete_paths.append(rec["full_path"])
            delete_info.append({
                "path": rec["path"],
                "filename": rec["filename"],
                "reason": rec["reason"]
            })
            
        # Write to JSON files
        with open("delete_paths.json", "w", encoding="utf-8") as f:
            json.dump(delete_paths, f, ensure_ascii=False, indent=2)
            
        with open("delete_info.json", "w", encoding="utf-8") as f:
            json.dump(delete_info, f, ensure_ascii=False, indent=2)
            
        # Add code to load from JSON files with "A" option for all deletions
        script_content += """
    # Process deletion recommendations
    print("\\nExecuting deletion recommendations...")
    
    try:
        if os.path.exists("delete_paths.json") and os.path.exists("delete_info.json"):
            with open("delete_paths.json", "r", encoding="utf-8") as f:
                PATHS_TO_DELETE = json.load(f)
                
            with open("delete_info.json", "r", encoding="utf-8") as f:
                DELETE_INFO = json.load(f)
            
            # Ask if user wants to apply to all
            auto_apply_all = False
            print("\\nFor deletions, you can:")
            print("  - Enter 'A' or 'a' to approve all deletions without further prompting")
            print("  - Enter 'y' or 'n' for each individual deletion")
                
            for i, (path, info) in enumerate(zip(PATHS_TO_DELETE, DELETE_INFO), 1):
                try:
                    if os.path.exists(path):
                        print(f"\\n{i}. Delete: {info['path']}")
                        print(f"   Reason: {info['reason']}")
                        
                        if not auto_apply_all:
                            response = input(f"   Proceed with deletion? (A:go all/y/n): ").lower()
                            if response == 'a':
                                auto_apply_all = True
                                proceed = True
                            else:
                                proceed = (response == 'y')
                        else:
                            proceed = True
                            print(f"   Auto-approving (all mode)...")
                        
                        if proceed:
                            os.remove(path)
                            log_action(log_file, "DELETED", info['path'])
                            print(f"   ✓ Deleted: {info['filename']}")
                        else:
                            print(f"   ✗ Skipped deletion")
                            log_action(log_file, "SKIPPED DELETE", info['path'])
                    else:
                        print(f"\\n{i}. File not found: {info['path']}")
                        log_action(log_file, "FILE NOT FOUND", info['path'])
                except Exception as e:
                    print(f"{i}. Error deleting {info['path']}: {e}")
                    log_action(log_file, "ERROR DELETING", info['path'], f"Error: {e}")
        else:
            print("No deletion data found. Skipping deletion operations.")
    except Exception as e:
        print(f"Error loading deletion data: {e}")
"""
    
    # Add rename actions using JSON files with "A" option for all renames
    if results["rename_recommendations"]:
        rename_paths = []
        rename_info = []
        
        for i, rec in enumerate(results["rename_recommendations"]):
            rename_paths.append({
                "old_path": rec["full_path"],
                "new_path": rec["new_full_path"]
            })
            rename_info.append({
                "path": rec["path"],
                "current_name": rec["current_name"],
                "suggested_name": rec["suggested_name"]
            })
            
        # Write to JSON files
        with open("rename_paths.json", "w", encoding="utf-8") as f:
            json.dump(rename_paths, f, ensure_ascii=False, indent=2)
            
        with open("rename_info.json", "w", encoding="utf-8") as f:
            json.dump(rename_info, f, ensure_ascii=False, indent=2)
            
        # Add code to load from JSON files with "A" option for all renames
        script_content += """
    # Process rename recommendations
    print("\\nExecuting rename recommendations...")
    
    try:
        if os.path.exists("rename_paths.json") and os.path.exists("rename_info.json"):
            with open("rename_paths.json", "r", encoding="utf-8") as f:
                PATHS_TO_RENAME = json.load(f)
                
            with open("rename_info.json", "r", encoding="utf-8") as f:
                RENAME_INFO = json.load(f)
            
            # Ask if user wants to apply to all
            auto_apply_all = False
            print("\\nFor renames, you can:")
            print("  - Enter 'A' or 'a' to approve all renames without further prompting")
            print("  - Enter 'y' or 'n' for each individual rename")
                
            for i, (paths, info) in enumerate(zip(PATHS_TO_RENAME, RENAME_INFO), 1):
                try:
                    if os.path.exists(paths['old_path']):
                        print(f"\\n{i}. Rename: {info['path']}")
                        print(f"   From: {info['current_name']}")
                        print(f"   To:   {info['suggested_name']}")
                        
                        if not auto_apply_all:
                            response = input(f"   Proceed with renaming? (A:go all/y/n): ").lower()
                            if response == 'a':
                                auto_apply_all = True
                                proceed = True
                            else:
                                proceed = (response == 'y')
                        else:
                            proceed = True
                            print(f"   Auto-approving (all mode)...")
                        
                        if proceed:
                            # Check if destination file already exists
                            if os.path.exists(paths['new_path']):
                                print(f"   ⚠ Destination file already exists!")
                                backup_response = "y"
                                if not auto_apply_all:
                                    backup_response = input(f"   Create backup of existing file before overwriting? (y/n): ").lower()
                                
                                if backup_response == 'y':
                                    backup_path = paths['new_path'] + f".backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    shutil.copy2(paths['new_path'], backup_path)
                                    BACKUP_FILES.append(backup_path)
                                    print(f"   ✓ Backup created: {os.path.basename(backup_path)}")
                                    log_action(log_file, "BACKUP CREATED", info['path'], f"backup: {os.path.basename(backup_path)}")
                                os.remove(paths['new_path'])
                                print(f"   ✓ Removed existing destination file")
                            
                            os.rename(paths['old_path'], paths['new_path'])
                            log_action(log_file, "RENAMED", info['path'], f"to: {info['suggested_name']}")
                            print(f"   ✓ Renamed successfully")
                        else:
                            print(f"   ✗ Skipped renaming")
                            log_action(log_file, "SKIPPED RENAME", info['path'])
                    else:
                        print(f"\\n{i}. File not found: {info['path']}")
                        log_action(log_file, "FILE NOT FOUND", info['path'])
                except Exception as e:
                    print(f"{i}. Error renaming {info['path']}: {e}")
                    log_action(log_file, "ERROR RENAMING", info['path'], f"Error: {e}")
        else:
            print("No rename data found. Skipping rename operations.")
    except Exception as e:
        print(f"Error loading rename data: {e}")
"""
    
    # Add backup file management and cleanup
    script_content += """
    # Find all backup files created during this session
    existing_backup_files = BACKUP_FILES
    
    # If no backup files created in this session, scan the directory tree
    if not existing_backup_files:
        print("\\nScanning for backup files created in previous sessions...")
        existing_backup_files = find_backup_files(ROOT_PATH)
    
    # Report backup files
    if existing_backup_files:
        print(f"\\nFound {len(existing_backup_files)} backup files:")
        for i, backup in enumerate(existing_backup_files, 1):
            created_time = os.path.getmtime(backup)
            created_str = datetime.datetime.fromtimestamp(created_time).strftime('%Y-%m-%d %H:%M:%S')
            size_mb = os.path.getsize(backup) / (1024 * 1024)
            print(f"  {i}. {os.path.basename(backup)}")
            print(f"     Created: {created_str}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Path: {os.path.dirname(backup)}")
            
        if input("\\nWould you like to delete all backup files? (y/n): ").lower() == 'y':
            for backup in existing_backup_files:
                try:
                    os.remove(backup)
                    log_action(log_file, "DELETED BACKUP", backup)
                    print(f"  ✓ Deleted backup: {os.path.basename(backup)}")
                except Exception as e:
                    print(f"  ✗ Error deleting backup {os.path.basename(backup)}: {e}")
                    log_action(log_file, "ERROR DELETING BACKUP", backup, f"Error: {e}")
            print(f"\\nBackup cleanup completed.")
        else:
            print(f"\\nBackup files preserved.")
    else:
        print("\\nNo backup files found.")
            
    # Optional: Clean up the JSON files
    try:
        for file in ["delete_paths.json", "delete_info.json", "rename_paths.json", "rename_info.json"]:
            if os.path.exists(file):
                os.remove(file)
    except Exception as e:
        print(f"Warning: Could not clean up JSON files: {e}")

    print("\\nAll operations completed!")
    print(f"See {log_file} for detailed log")

if __name__ == "__main__":
    main()
"""
    
    # Write the script with UTF-8 encoding
    with open(script_filename, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print(f"\nAction script generated: {script_filename}")
    print(f"Run this script to execute the recommended actions")

def main():
    # Get user input for root directory
    root_directory = input("Enter the root directory to analyze: ")
    
    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist.")
        return
    
    if not os.path.isdir(root_directory):
        print(f"Error: '{root_directory}' is not a directory.")
        return
    
    results = analyze_directory_structure(root_directory)
    generate_todo_list(results)
    
    if results["delete_recommendations"] or results["rename_recommendations"]:
        if input("\nWould you like to generate an action script to execute these recommendations? (y/n): ").lower() == 'y':
            generate_action_script(results, root_directory)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()