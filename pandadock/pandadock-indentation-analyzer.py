#!/usr/bin/env python3
"""
PandaDock Indentation Analyzer

This script is specifically designed to analyze indentation issues in the PandaDock codebase.
It identifies common indentation problems and provides guidance on how to fix them.

Usage:
    python pandadock_indentation_analyzer.py pandadock_file.py

Or save your code to a file first and then analyze it:
    python pandadock_indentation_analyzer.py path/to/saved_code.py
"""

import sys
import os
import re
from collections import defaultdict

def analyze_indentation(filename):
    """Analyze indentation issues in a file."""
    print(f"Analyzing indentation in {filename}...")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Analysis structures
    issues = []
    function_blocks = {}
    current_block = None
    block_start = 0
    indentation_stack = []
    current_indent = 0
    problematic_functions = set()
    
    # Track line continuations and open brackets
    continuations = []
    open_brackets = []
    
    # First pass: identify function blocks and basic indentation
    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
            
        # Calculate indentation
        indent = len(line) - len(line.lstrip())
        
        # Track line continuations
        if stripped.endswith('\\'):
            continuations.append(line_num)
            
        # Track bracket balance
        open_count = stripped.count('(') + stripped.count('[') + stripped.count('{')
        close_count = stripped.count(')') + stripped.count(']') + stripped.count('}')
        
        if open_count > close_count:
            open_brackets.append(line_num)
        elif close_count > open_count and open_brackets:
            open_brackets.pop()
        
        # Function/method definition
        if re.match(r'^\s*def\s+\w+\s*\(', line):
            func_name = re.search(r'def\s+(\w+)', stripped).group(1)
            
            # End previous block if any
            if current_block:
                function_blocks[current_block] = (block_start, line_num - 1)
                
            current_block = func_name
            block_start = line_num
            indentation_stack = [indent]
            current_indent = indent
            
        # Class definition
        elif re.match(r'^\s*class\s+\w+', line):
            class_name = re.search(r'class\s+(\w+)', stripped).group(1)
            
            # End previous block if any
            if current_block:
                function_blocks[current_block] = (block_start, line_num - 1)
                
            current_block = class_name
            block_start = line_num
            indentation_stack = [indent]
            current_indent = indent
            
        # End of block detection based on indentation
        elif current_block and indent <= indentation_stack[0] and not line_num - 1 in continuations and not line_num - 1 in open_brackets:
            function_blocks[current_block] = (block_start, line_num - 1)
            current_block = None
        
        # Track indentation changes within blocks
        if current_block:
            if indent > current_indent:
                # Indentation increase
                if indent - current_indent != 4:
                    issues.append({
                        'line': line_num,
                        'type': 'inconsistent_indent',
                        'message': f"Inconsistent indentation increase: {indent - current_indent} spaces instead of 4",
                        'block': current_block
                    })
                    problematic_functions.add(current_block)
                
                indentation_stack.append(indent)
                current_indent = indent
            
            elif indent < current_indent:
                # Dedent - should match a previous level
                while indentation_stack and current_indent > indent:
                    indentation_stack.pop()
                    if indentation_stack:
                        current_indent = indentation_stack[-1]
                
                if not indentation_stack or indent != current_indent:
                    issues.append({
                        'line': line_num,
                        'type': 'misaligned_dedent',
                        'message': f"Dedent to {indent} spaces doesn't match any previous indentation level",
                        'block': current_block
                    })
                    problematic_functions.add(current_block)
            
            # Check if line after colon is properly indented
            if line.rstrip().endswith(':') and i+1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line and not next_line.startswith('#'):
                    next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                    if next_indent <= indent:
                        issues.append({
                            'line': line_num + 1,
                            'type': 'missing_indent',
                            'message': f"Missing indentation after line ending with colon",
                            'block': current_block
                        })
                        problematic_functions.add(current_block)
    
    # End the last block if necessary
    if current_block:
        function_blocks[current_block] = (block_start, len(lines))
    
    # Second pass: Look for specific issues in known problematic functions
    target_functions = [
        'prepare_protein_configs', 
        '_detect_flexible_residues', 
        'main',
        'write_results_to_txt'
    ]
    
    # Add identified problematic functions to the list
    target_functions.extend(list(problematic_functions))
    
    # Make the list unique
    target_functions = list(set(target_functions))
    
    for func in target_functions:
        if func in function_blocks:
            start, end = function_blocks[func]
            func_lines = lines[start-1:end]
            
            # Analyze this function in detail
            analyze_function_indentation(func, func_lines, start, issues)
    
    # Report findings
    print_indent_report(issues, lines, function_blocks)
    
    return issues, function_blocks, lines
    

def analyze_function_indentation(func_name, lines, start_line, issues):
    """Perform detailed analysis of a function's indentation."""
    # Track indentation patterns and inconsistencies
    indent_patterns = {}
    common_issues = {
        'inconsistent_base': False,
        'mixed_indent_size': False,
        'complex_conditionals': []
    }
    
    # Extract the base indentation from the function definition
    base_indent = len(lines[0]) - len(lines[0].lstrip())
    
    # Track if/else blocks
    if_blocks = []
    current_if_block = None
    
    for i, line in enumerate(lines):
        line_num = start_line + i
        stripped = line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue
            
        indent = len(line) - len(line.lstrip())
        relative_indent = indent - base_indent
        
        # Track indentation sizes
        if relative_indent > 0:
            level = relative_indent // 4
            remainder = relative_indent % 4
            
            if remainder != 0:
                issues.append({
                    'line': line_num,
                    'type': 'non_std_indent',
                    'message': f"Non-standard indentation of {relative_indent} spaces (not a multiple of 4)",
                    'block': func_name
                })
                common_issues['mixed_indent_size'] = True
            
            if level not in indent_patterns:
                indent_patterns[level] = 0
            indent_patterns[level] += 1
        
        # Track if/elif/else structures
        if re.search(r'^\s*if\s+', line):
            if current_if_block:
                if_blocks.append(current_if_block)
            current_if_block = {
                'start': line_num,
                'indent': indent,
                'has_else': False,
                'lines': []
            }
        
        elif re.search(r'^\s*elif\s+', line) and current_if_block:
            if indent != current_if_block['indent']:
                issues.append({
                    'line': line_num,
                    'type': 'misaligned_control',
                    'message': f"elif statement indentation doesn't match if statement at line {current_if_block['start']}",
                    'block': func_name
                })
        
        elif re.search(r'^\s*else\s*:', line) and current_if_block:
            current_if_block['has_else'] = True
            if indent != current_if_block['indent']:
                issues.append({
                    'line': line_num,
                    'type': 'misaligned_control',
                    'message': f"else statement indentation doesn't match if statement at line {current_if_block['start']}",
                    'block': func_name
                })
        
        # Look for complex conditional expressions
        if 'if ' in stripped or 'elif ' in stripped:
            if stripped.count('and ') > 1 or stripped.count('or ') > 1:
                common_issues['complex_conditionals'].append(line_num)
        
        # Track indentation after context manager blocks
        if 'with ' in stripped and stripped.rstrip().endswith(':'):
            # Check next line indentation
            if i + 1 < len(lines):
                next_indent = len(lines[i+1]) - len(lines[i+1].lstrip())
                if next_indent <= indent:
                    issues.append({
                        'line': line_num + 1,
                        'type': 'missing_indent',
                        'message': f"Missing indentation after 'with' statement",
                        'block': func_name
                    })
        
        # End the current if block if we dedent past it
        if current_if_block and indent < current_if_block['indent']:
            if_blocks.append(current_if_block)
            current_if_block = None
            
            # Check if there's a parent if block we're still in
            parent_blocks = [b for b in if_blocks if b['indent'] < indent]
            if parent_blocks:
                current_if_block = parent_blocks[-1]
                if_blocks.remove(current_if_block)
        
        # Record line in current if block if any
        if current_if_block:
            current_if_block['lines'].append(line_num)
    
    # Add the last if block if any
    if current_if_block:
        if_blocks.append(current_if_block)
    
    # Check for inconsistent indentation in if blocks
    for block in if_blocks:
        if block['has_else'] and len(block['lines']) > 4:
            # Find all unique indentation levels in this block
            block_indents = set()
            for line_num in block['lines']:
                idx = line_num - start_line
                if idx < len(lines):
                    line = lines[idx]
                    indent = len(line) - len(line.lstrip())
                    block_indents.add(indent)
            
            # If more than base + 1 indentation level, might be complex
            if len(block_indents) > 2:
                issues.append({
                    'line': block['start'],
                    'type': 'complex_if_block',
                    'message': f"Complex if/else block with multiple indentation levels",
                    'block': func_name
                })
    
    return issues


def print_indent_report(issues, lines, function_blocks):
    """Print a detailed report of indentation issues."""
    if not issues:
        print("No indentation issues found.")
        return
    
    print(f"\n===== Found {len(issues)} indentation issues =====")
    
    # Group issues by type
    issue_types = defaultdict(list)
    for issue in issues:
        issue_types[issue['type']].append(issue)
    
    # Group issues by function
    issues_by_function = defaultdict(list)
    for issue in issues:
        if 'block' in issue:
            issues_by_function[issue['block']].append(issue)
    
    # Print summary by type
    print("\n=== Issues by Type ===")
    for issue_type, type_issues in issue_types.items():
        print(f"{issue_type.replace('_', ' ').title()}: {len(type_issues)} issues")
    
    # Print summary by function
    print("\n=== Issues by Function ===")
    for func, func_issues in issues_by_function.items():
        if func in function_blocks:
            start, end = function_blocks[func]
            print(f"{func} (lines {start}-{end}): {len(func_issues)} issues")
    
    # Print top issues with context
    print("\n=== Top Issues ===")
    for i, issue in enumerate(sorted(issues, key=lambda x: x['line'])):
        if i >= 20:  # Limit to top 20 issues
            print(f"... and {len(issues) - 20} more issues")
            break
            
        line_num = issue['line']
        print(f"Line {line_num}: {issue['message']}")
        
        # Show context (the line with the issue and possibly surrounding lines)
        if 0 <= line_num - 1 < len(lines):
            context_start = max(0, line_num - 2)
            context_end = min(len(lines), line_num + 1)
            
            for j in range(context_start, context_end):
                line_marker = "→ " if j == line_num - 1 else "  "
                print(f"{line_marker}{j+1}: {lines[j].rstrip()}")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if 'inconsistent_indent' in issue_types:
        print("• Fix inconsistent indentation increments:")
        print("  - Always use 4 spaces per indentation level")
        print("  - Verify all editor settings use consistent tab/space settings")
    
    if 'misaligned_dedent' in issue_types:
        print("• Fix misaligned dedents:")
        print("  - Ensure all dedents align with a previous indentation level")
        print("  - Check for mixed tabs and spaces that might cause alignment issues")
    
    if 'misaligned_control' in issue_types:
        print("• Fix misaligned control structures:")
        print("  - Ensure all elif/else statements align with their corresponding if statement")
    
    if 'missing_indent' in issue_types:
        print("• Fix missing indentation after blocks:")
        print("  - Add proper indentation after lines ending with a colon")
    
    # Functions most needing attention
    print("\n=== Functions Needing Most Attention ===")
    for func, func_issues in sorted(issues_by_function.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
        print(f"{func}: {len(func_issues)} issues")
    
    print("\n=== Next Steps ===")
    print("1. Start by fixing issues in the most problematic functions")
    print("2. Use a consistent indentation style throughout (4 spaces per level recommended)")
    print("3. Consider using an automatic formatter like 'black' or 'autopep8'")
    print("4. Configure your editor with proper Python indentation settings")
    print("5. Run a PEP 8 linter to catch other style issues")


def fix_common_issues(filename, output_filename=None):
    """Attempt to fix common indentation issues."""
    if output_filename is None:
        output_filename = filename + '.fixed.py'
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Analyze to find issues
    issues, function_blocks, _ = analyze_indentation(filename)
    
    # Group issues by line
    issues_by_line = defaultdict(list)
    for issue in issues:
        issues_by_line[issue['line']].append(issue)
    
    # Fix common indentation issues
    fixed_lines = lines.copy()
    fixes_applied = 0
    
    for line_num, line_issues in issues_by_line.items():
        if line_num <= 0 or line_num > len(fixed_lines):
            continue
            
        idx = line_num - 1
        line = fixed_lines[idx]
        
        # Main issue types to fix
        issue_types = [issue['type'] for issue in line_issues]
        
        # Fix inconsistent indentation
        if 'inconsistent_indent' in issue_types or 'non_std_indent' in issue_types:
            # Calculate what indentation should be
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Find the function this line belongs to
            containing_function = None
            for func, (start, end) in function_blocks.items():
                if start <= line_num <= end:
                    containing_function = func
                    break
            
            # If we found the function, determine correct indentation
            if containing_function:
                func_start, func_end = function_blocks[containing_function]
                func_base_indent = len(fixed_lines[func_start-1]) - len(fixed_lines[func_start-1].lstrip())
                
                # Try to determine correct level by looking at surrounding lines
                correct_level = 0
                
                # Look at previous lines to find the indentation pattern
                for i in range(idx-1, max(0, func_start-2), -1):
                    prev_line = fixed_lines[i]
                    prev_stripped = prev_line.strip()
                    
                    if not prev_stripped or prev_stripped.startswith('#'):
                        continue
                        
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    
                    # If previous line ends with colon, this line should be indented one more level
                    if prev_line.rstrip().endswith(':'):
                        correct_level = (prev_indent - func_base_indent) // 4 + 1
                        break
                    
                    # If previous line has same or higher indentation, use the same level
                    if prev_indent >= current_indent:
                        correct_level = (prev_indent - func_base_indent) // 4
                        break
                        
                    # If previous line has lower indentation, check if it matches a control structure
                    if prev_indent < current_indent:
                        # Find nearest if/elif/else or other block structure
                        if re.search(r'^\s*(if|elif|else|for|while|def|class|with|try|except|finally)\b', prev_line):
                            correct_level = (prev_indent - func_base_indent) // 4 + 1
                            break
                
                # Calculate correct indentation
                correct_indent = func_base_indent + (correct_level * 4)
                
                # Only fix if we're confident
                if correct_indent != current_indent:
                    fixed_lines[idx] = ' ' * correct_indent + stripped
                    fixes_applied += 1
        
        # Fix misaligned dedent
        elif 'misaligned_dedent' in issue_types:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Look at previous lines to find the correct dedent level
            for i in range(idx-1, -1, -1):
                prev_line = fixed_lines[i]
                prev_stripped = prev_line.strip()
                
                if not prev_stripped or prev_stripped.startswith('#'):
                    continue
                    
                prev_indent = len(prev_line) - len(prev_line.lstrip())
                
                # If we find a line with less indentation than current, use that
                if prev_indent < current_indent:
                    fixed_lines[idx] = ' ' * prev_indent + stripped
                    fixes_applied += 1
                    break
    
    # Write fixed file
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)
    
    print(f"\nFixed {fixes_applied} indentation issues. Output saved to {output_filename}")
    return fixes_applied


def fix_pasted_code(code_text, output_filename="fixed_code.py"):
    """Fix indentation in pasted code."""
    # Write to temporary file
    temp_file = "temp_code_to_fix.py"
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(code_text)
    
    # Call the fix function
    fixes = fix_common_issues(temp_file, output_filename)
    
    # Clean up temp file
    try:
        os.remove(temp_file)
    except:
        pass
    
    return fixes


def main():
    """Main function for the PandaDock indentation analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python pandadock_indentation_analyzer.py [filepath or paste_file.txt]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} does not exist.")
        sys.exit(1)
    
    # Check if the file is a Python file or pasted code
    if input_file.endswith('.py'):
        # Analyze and print report
        analyze_indentation(input_file)
        
        # Ask if user wants to fix issues
        response = input("\nWould you like to attempt to fix these issues? (y/n): ")
        if response.lower() in ('y', 'yes'):
            output_file = input("Enter output filename (leave empty for [original].fixed.py): ")
            if not output_file:
                output_file = input_file + '.fixed.py'
            
            fix_common_issues(input_file, output_file)
    else:
        # Assume it's pasted code
        with open(input_file, 'r', encoding='utf-8') as f:
            code_text = f.read()
        
        # Write to a temporary Python file
        temp_file = "temp_pandadock_code.py"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(code_text)
        
        # Analyze the temporary file
        analyze_indentation(temp_file)
        
        # Ask if user wants to fix issues
        response = input("\nWould you like to attempt to fix these issues? (y/n): ")
        if response.lower() in ('y', 'yes'):
            output_file = input("Enter output filename (leave empty for fixed_pandadock_code.py): ")
            if not output_file:
                output_file = "fixed_pandadock_code.py"
            
            fix_common_issues(temp_file, output_file)
        
        # Clean up temp file
        try:
            os.remove(temp_file)
        except:
            pass


if __name__ == "__main__":
    main()