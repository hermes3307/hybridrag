import os
import re
import shutil
import datetime
import json
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_filename_features(filenames):
    """Extract features from filenames for vector similarity search"""
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',  # Character n-grams for better Korean handling
        ngram_range=(2, 4),  # Use 2-4 character n-grams
        max_features=1000
    )
    features = vectorizer.fit_transform(filenames)
    return features, vectorizer

def find_similar_filenames(target_filename, filenames, features):
    """Find similar filenames using vector similarity"""
    target_vector = features[filenames.index(target_filename)]
    similarities = cosine_similarity(target_vector, features).flatten()
    similar_indices = np.argsort(similarities)[::-1]
    similar_indices = [idx for idx in similar_indices if filenames[idx] != target_filename]
    
    results = []
    for idx in similar_indices[:5]:
        results.append({
            "filename": filenames[idx],
            "similarity": similarities[idx]
        })
    return results

def recommend_standardized_filename(filename, similar_files):
    """Generate standardized filename recommendations"""
    base, ext = os.path.splitext(filename)
    recommendations = []
    
    # Rule 1: Replace spaces with underscores
    if ' ' in base:
        recommendations.append({
            "name": re.sub(r'\s+', '_', base) + ext,
            "reason": "Replace spaces with underscores"
        })
    
    # Rule 2: Remove special characters except underscores and hyphens
    cleaned_name = re.sub(r'[^\w\s\-가-힣]', '', base) + ext
    if cleaned_name != filename:
        recommendations.append({
            "name": cleaned_name,
            "reason": "Remove special characters"
        })
    
    # Rule 3: Use information from similar files if available
    if similar_files and similar_files[0]["similarity"] > 0.7:
        ref_base = os.path.splitext(similar_files[0]["filename"])[0]
        
        # Extract patterns like dates, etc. from the reference file
        date_pattern = r'(\d{4}[-_.]\d{2}[-_.]\d{2}|\d{2}[-_.]\d{2}[-_.]\d{4}|\d{8})'
        ref_date_match = re.search(date_pattern, ref_base)
        
        if ref_date_match:
            # If the reference has a date pattern and the current file doesn't
            if not re.search(date_pattern, base):
                date_str = ref_date_match.group(1)
                recommendations.append({
                    "name": f"{base}_{date_str}{ext}",
                    "reason": f"Add date pattern found in similar files ({date_str})"
                })
    
    return recommendations

def standardize_filenames(root_path, backup_dir="./backup_files"):
    """Standardize filenames across the directory structure"""
    print(f"Starting filename standardization for: {root_path}")
    print("-" * 60)
    
    # Create backup directory
    backup_dir = os.path.join(root_path, backup_dir)
    os.makedirs(backup_dir, exist_ok=True)
    
    # Collect all filenames
    all_files = []
    file_paths = {}
    
    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            all_files.append(filename)
            file_paths[filename] = os.path.join(dirpath, filename)
    
    if not all_files:
        print("No files found to standardize.")
        return
    
    print(f"Found {len(all_files)} files to analyze")
    
    # Extract features for similarity search
    try:
        features, _ = extract_filename_features(all_files)
        print("Successfully extracted filename features")
    except Exception as e:
        print(f"Error extracting filename features: {e}")
        features = None
    
    # Process each file
    standardization_actions = []
    
    for i, filename in enumerate(all_files, 1):
        print(f"\nAnalyzing [{i}/{len(all_files)}]: {filename}")
        
        # Find similar filenames
        similar_files = []
        if features is not None:
            try:
                similar_files = find_similar_filenames(filename, all_files, features)
                if similar_files:
                    print(f"Found {len(similar_files)} similar files")
            except Exception as e:
                print(f"Error finding similar filenames: {e}")
        
        # Get standardization recommendations
        recommendations = recommend_standardized_filename(filename, similar_files)
        
        if recommendations:
            print("Recommendations:")
            for j, rec in enumerate(recommendations, 1):
                print(f"  {j}. {rec['name']} ({rec['reason']})")
            
            # Add to actions list
            filepath = file_paths[filename]
            dirpath, _ = os.path.split(filepath)
            
            for rec in recommendations:
                standardization_actions.append({
                    "original_path": filepath,
                    "suggested_name": rec["name"],
                    "new_path": os.path.join(dirpath, rec["name"]),
                    "reason": rec["reason"],
                    "backup_path": os.path.join(backup_dir, filename)
                })
        else:
            print("No standardization recommendations for this file")
    
    # Generate action script for standardization
    if standardization_actions:
        print(f"\nGenerated {len(standardization_actions)} recommendations")
        
        # Create standardization script
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        script_filename = f"execute_standardization_{timestamp}.py"
        
        with open(script_filename, "w", encoding="utf-8") as f:
            f.write("""# -*- coding: utf-8 -*-
import os
import shutil
import datetime
import json
import sys

# Ensure proper encoding for console output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

def log_action(log_file, action, path, details=""):
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{timestamp} - {action}: {path} {details}\\n")

def main():
    # Create log file
    log_file = f"standardization_actions_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Filename Standardization Log - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
        f.write("-" * 80 + "\\n")
    
    print("Starting standardization of filenames...")
    print(f"Actions will be logged to {log_file}")
    
    # Load standardization recommendations
    try:
        with open("standardization_actions.json", "r", encoding="utf-8") as f:
            standardization_actions = json.load(f)
        
        # Group recommendations by file
        actions_by_file = {}
        for action in standardization_actions:
            orig_path = action['original_path']
            if orig_path not in actions_by_file:
                actions_by_file[orig_path] = []
            actions_by_file[orig_path].append(action)
        
        print(f"\\nFound {len(actions_by_file)} files with recommendations")
        
        # Ask if user wants to apply to all files automatically
        print("\\nFor standardization, you can:")
        print("  - Enter 'A' or 'a' to automatically apply first recommendation to all files")
        print("  - Enter 'y' to proceed with individual selections")
        print("  - Enter any other key to exit")
        
        auto_mode = input("\\nApply first recommendation to all files? (A/y/n): ").lower()
        
        if auto_mode not in ['a', 'y']:
            print("Exiting standardization process.")
            return
        
        # Process each file
        for i, (orig_path, actions) in enumerate(actions_by_file.items(), 1):
            if not os.path.exists(orig_path):
                print(f"\\n{i}. File not found: {orig_path}")
                log_action(log_file, "FILE NOT FOUND", orig_path)
                continue
                
            filename = os.path.basename(orig_path)
            print(f"\\n{i}. File: {filename}")
            
            # Select action based on mode
            if auto_mode == 'a':
                # Automatically apply first recommendation
                selected_action = actions[0]
                print(f"   Auto-applying: {selected_action['suggested_name']}")
                print(f"   Reason: {selected_action['reason']}")
            else:
                # Show recommendations
                print(f"   {len(actions)} recommendations:")
                for j, action in enumerate(actions, 1):
                    print(f"     {j}. {action['suggested_name']}")
                    print(f"        Reason: {action['reason']}")
                
                # Ask user which recommendation to apply
                choice = input(f"\\n   Which recommendation? (1-{len(actions)}, s to skip, A for all remaining): ")
                
                if choice.lower() == 's':
                    print("   ✗ Skipped")
                    log_action(log_file, "SKIPPED", orig_path)
                    continue
                elif choice.lower() == 'a':
                    # User chose to apply all remaining automatically
                    auto_mode = 'a'
                    selected_action = actions[0]
                    print(f"   Auto-applying: {selected_action['suggested_name']}")
                else:
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(actions):
                            selected_action = actions[choice_idx]
                        else:
                            print(f"   ✗ Invalid choice. Skipping.")
                            log_action(log_file, "SKIPPED", orig_path, "Invalid choice")
                            continue
                    except ValueError:
                        print(f"   ✗ Invalid input. Skipping.")
                        log_action(log_file, "SKIPPED", orig_path, "Invalid input")
                        continue
            
            # Process the selected action
            new_path = selected_action['new_path']
            backup_path = selected_action['backup_path']
            
            # Check if destination already exists
            if os.path.exists(new_path) and new_path != orig_path:
                print(f"   ⚠ Destination file already exists!")
                
                if auto_mode == 'a':
                    # In auto mode, always create backup
                    backup_file = True
                else:
                    backup_file = input(f"   Create backup of existing file before overwriting? (y/n): ").lower() == 'y'
                
                if backup_file:
                    dest_backup_path = new_path + f".backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy2(new_path, dest_backup_path)
                    print(f"   ✓ Destination backup created")
                    log_action(log_file, "DESTINATION BACKUP", new_path, f"to: {dest_backup_path}")
                
                # Remove existing destination file
                os.remove(new_path)
                print(f"   ✓ Removed existing destination file")
            
            # Create backup of original file
            backup_dir = os.path.dirname(backup_path)
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(orig_path, backup_path)
            print(f"   ✓ Created original backup")
            log_action(log_file, "BACKUP CREATED", orig_path, f"to: {backup_path}")
            
            # Rename file
            if new_path != orig_path:  # Only rename if different
                os.rename(orig_path, new_path)
                print(f"   ✓ Renamed to: {os.path.basename(new_path)}")
                log_action(log_file, "RENAMED", orig_path, f"to: {new_path}")
            else:
                print("   ℹ New name is the same as current name")
                log_action(log_file, "NO CHANGE", orig_path)
    
    except Exception as e:
        print(f"Error processing standardization actions: {e}")
        if log_file:
            log_action(log_file, "ERROR", "", f"Error: {e}")
    
    # Clean up
    try:
        if os.path.exists("standardization_actions.json"):
            os.remove("standardization_actions.json")
    except Exception as e:
        print(f"Warning: Could not clean up JSON files: {e}")
    
    print("\\nStandardization completed!")
    print(f"See {log_file} for detailed log")

if __name__ == "__main__":
    main()
""")
        
        # Save standardization actions to JSON
        with open("standardization_actions.json", "w", encoding="utf-8") as f:
            json.dump(standardization_actions, f, ensure_ascii=False, indent=2)
            
        print(f"\nStandardization script generated: {script_filename}")
        print(f"Run this script to execute the filename standardization")
    else:
        print("\nNo standardization recommendations generated.")

if __name__ == "__main__":
    root_directory = input("Enter the root directory to standardize: ")
    
    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist.")
    elif not os.path.isdir(root_directory):
        print(f"Error: '{root_directory}' is not a directory.")
    else:
        standardize_filenames(root_directory)