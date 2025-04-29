#!/usr/bin/env python3
"""
Indentation Linter for Python Code

This script analyzes Python files for indentation issues and provides detailed reports
on potential problems. It's specifically designed to help with PandaDock codebase.

Usage:
    python indentation_linter.py [filepath]

Example:
    python indentation_linter.py path/to/your/python_file.py
"""

import sys
import os
import re
import tokenize
import io
from collections import defaultdict


class IndentationLinter:
    def __init__(self, tab_size=4):
        self.tab_size = tab_size
        self.issues = []
        self.indentation_levels = []
        self.current_indentation = 0
        self.line_continuations = set()
        self.bracket_stack = []
        self.consistent_indentation = None  # None until determined
        self.function_defs = {}  # Track function definitions and their indentation
        self.class_defs = {}     # Track class definitions and their indentation
        self.nested_blocks = []  # Stack to track nested blocks (if, for, while, etc.)

    def check_file(self, filepath):
        """Check a file for indentation issues."""
        print(f"Analyzing {filepath}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return self._check_content(content, filepath)
        except Exception as e:
            print(f"Error analyzing file {filepath}: {str(e)}")
            return []

    def _check_content(self, content, filepath):
        """Check content string for indentation issues."""
        self.issues = []
        self.indentation_levels = []
        self.current_indentation = 0
        self.line_continuations = set()
        self.bracket_stack = []
        self.consistent_indentation = None
        self.function_defs = {}
        self.class_defs = {}
        self.nested_blocks = []
        
        # Find line continuations first (lines ending with \)
        lines = content.split('\n')
        for line_no, line in enumerate(lines, 1):
            line = line.rstrip()
            if line.endswith('\\'):
                self.line_continuations.add(line_no)
        
        # Find open brackets for implicit line continuations
        open_brackets = defaultdict(list)
        for token_info in tokenize.tokenize(io.BytesIO(content.encode('utf-8')).readline):
            if token_info.type == tokenize.OP and token_info.string in ('(', '[', '{'):
                open_brackets[token_info.start[0]].append((token_info.string, token_info.start))
            elif token_info.type == tokenize.OP and token_info.string in (')', ']', '}'):
                if token_info.start[0] in open_brackets and open_brackets[token_info.start[0]]:
                    open_brackets[token_info.start[0]].pop()
        
        # Now check line by line for indentation issues
        for line_no, line in enumerate(lines, 1):
            stripped_line = line.lstrip()
            
            # Skip empty lines or comment-only lines
            if not stripped_line or stripped_line.startswith('#'):
                continue
            
            # Calculate indentation level
            indentation = len(line) - len(stripped_line)
            
            # Check if this line should be indented due to previous line
            self._check_expected_indentation(line_no, indentation, stripped_line, lines, open_brackets)
            
            # Track function and class definitions
            if stripped_line.startswith('def '):
                self.function_defs[line_no] = indentation
            elif stripped_line.startswith('class '):
                self.class_defs[line_no] = indentation
            
            # Check for inconsistent indentation units
            self._check_indentation_consistency(indentation, line_no)
            
            # Check for mixed tabs and spaces
            if '\t' in line[:indentation] and ' ' in line[:indentation]:
                self.issues.append({
                    'line': line_no,
                    'type': 'mixed_tabs_spaces',
                    'message': 'Line uses both tabs and spaces for indentation'
                })
            
            # Update indentation tracking
            self.indentation_levels.append(indentation)
            self.current_indentation = indentation

        # Final report
        if self.issues:
            print(f"Found {len(self.issues)} potential indentation issues in {filepath}")
        else:
            print(f"No indentation issues found in {filepath}")
        
        return self.issues

    def _check_expected_indentation(self, line_no, indentation, stripped_line, lines, open_brackets):
        """Check if the line's indentation matches what we would expect."""
        # Check if previous line ended with a colon (expecting indentation)
        if line_no > 1:
            prev_line = lines[line_no - 2].rstrip()
            
            # Check if line is a continuation (implicit or explicit)
            is_continuation = (line_no - 1) in self.line_continuations or \
                             (line_no - 1) in open_brackets and open_brackets[line_no - 1]
            
            if prev_line.endswith(':') and not is_continuation:
                # This line should be indented
                prev_indent = len(lines[line_no - 2]) - len(lines[line_no - 2].lstrip())
                expected_indent = prev_indent + self.tab_size
                
                if indentation <= prev_indent:
                    self.issues.append({
                        'line': line_no,
                        'type': 'missing_indentation',
                        'message': f'Expected indentation of at least {expected_indent} spaces after block statement'
                    })
            
            # Check for dedent - should align with a previous indentation level
            elif indentation < self.current_indentation and indentation not in self.indentation_levels:
                self.issues.append({
                    'line': line_no,
                    'type': 'misaligned_dedent',
                    'message': f'Dedent does not match any previous indentation level'
                })

    def _check_indentation_consistency(self, indentation, line_no):
        """Check if indentation is consistent with the established pattern."""
        if indentation > 0:
            if self.consistent_indentation is None:
                # First indented line - establish the pattern
                if indentation % self.tab_size == 0:
                    self.consistent_indentation = self.tab_size
                else:
                    self.consistent_indentation = indentation
                    if indentation != self.tab_size:
                        self.issues.append({
                            'line': line_no,
                            'type': 'unusual_indentation',
                            'message': f'First indentation is {indentation} spaces, not {self.tab_size} as expected'
                        })
            elif indentation % self.consistent_indentation != 0:
                self.issues.append({
                    'line': line_no,
                    'type': 'inconsistent_indentation',
                    'message': f'Indentation of {indentation} spaces is not a multiple of the established {self.consistent_indentation}'
                })

    def print_report(self):
        """Print a detailed report of all indentation issues."""
        if not self.issues:
            print("No indentation issues found.")
            return
        
        print("\n===== Indentation Issues Report =====")
        
        # Group issues by type
        issue_types = defaultdict(list)
        for issue in self.issues:
            issue_types[issue['type']].append(issue)
        
        for issue_type, issues in issue_types.items():
            print(f"\n{issue_type.replace('_', ' ').title()} ({len(issues)} issues):")
            for issue in issues:
                print(f"  Line {issue['line']}: {issue['message']}")
        
        print("\n=== Summary ===")
        print(f"Total issues: {len(self.issues)}")
        print(f"Types of issues: {', '.join(issue_types.keys())}")
        
        # Provide recommendations
        print("\n=== Recommendations ===")
        if 'mixed_tabs_spaces' in issue_types:
            print("- Convert all indentation to spaces (recommended: 4 spaces per level)")
        if 'inconsistent_indentation' in issue_types:
            print(f"- Standardize on {self.consistent_indentation or self.tab_size} spaces per indentation level")
        if 'missing_indentation' in issue_types:
            print("- Check all code blocks after colons (:) for proper indentation")
        if 'misaligned_dedent' in issue_types:
            print("- Ensure all dedents align with a previous indentation level")


def analyze_directory(directory_path, extensions=None):
    """Analyze all Python files in a directory."""
    if extensions is None:
        extensions = ['.py']
    
    linter = IndentationLinter()
    all_issues = []
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, file)
                issues = linter.check_file(filepath)
                if issues:
                    all_issues.append((filepath, issues))
    
    return all_issues


def find_specific_issues(content, target_functions=None):
    """Find potential issues in specific functions of interest."""
    if target_functions is None:
        target_functions = [
            'prepare_protein_configs', 
            '_detect_flexible_residues', 
            'MonteCarloSampling',
            'main'
        ]
    
    issues = []
    lines = content.split('\n')
    
    # Find function definitions and their blocks
    function_blocks = {}
    current_function = None
    function_start = 0
    base_indent = 0
    
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#'):
            continue
        
        indent = len(line) - len(stripped)
        
        # Check for function definition
        if stripped.startswith('def '):
            func_name = stripped[4:].split('(')[0].strip()
            
            # If we were tracking a function, finish it
            if current_function:
                function_blocks[current_function] = (function_start, i - 1)
            
            # Start tracking new function
            current_function = func_name
            function_start = i
            base_indent = indent
        
        # If we're outside the base indentation of a function, we're done with it
        elif current_function and indent <= base_indent and not stripped.startswith('#') and stripped:
            function_blocks[current_function] = (function_start, i - 1)
            current_function = None
    
    # Add the last function if needed
    if current_function:
        function_blocks[current_function] = (function_start, len(lines) - 1)
    
    # Now analyze the target functions
    for func in target_functions:
        if func in function_blocks:
            start, end = function_blocks[func]
            func_lines = lines[start:end+1]
            
            # Check for issues
            indent_stack = []
            current_indent = None
            
            for i, line in enumerate(func_lines, start + 1):
                stripped = line.lstrip()
                if not stripped or stripped.startswith('#'):
                    continue
                
                indent = len(line) - len(stripped)
                
                if current_indent is None:
                    current_indent = indent
                
                # Track indentation changes
                if indent > current_indent:
                    indent_stack.append(current_indent)
                    
                    # Check if indentation increment is consistent
                    if indent - current_indent != 4:
                        issues.append({
                            'line': i,
                            'function': func,
                            'message': f'Unusual indentation increment: {indent - current_indent} spaces (expected 4)'
                        })
                    
                    current_indent = indent
                elif indent < current_indent:
                    # Check dedent
                    while indent_stack and current_indent > indent:
                        current_indent = indent_stack.pop()
                    
                    if current_indent != indent:
                        issues.append({
                            'line': i,
                            'function': func,
                            'message': f'Dedent to {indent} spaces does not match any previous indentation level'
                        })
    
    return issues


def main():
    """Main function for the indentation checker."""
    if len(sys.argv) < 2:
        print("Usage: python indentation_linter.py [filepath or directory]")
        return
    
    path = sys.argv[1]
    
    if os.path.isdir(path):
        all_issues = analyze_directory(path)
        if all_issues:
            print(f"\nFound indentation issues in {len(all_issues)} files.")
            for filepath, issues in all_issues:
                print(f"\n{filepath}: {len(issues)} issues")
        else:
            print("No indentation issues found in any files.")
    else:
        if not os.path.exists(path):
            print(f"Error: {path} does not exist.")
            return
            
        linter = IndentationLinter()
        
        # If it's a Python file, do standard checking
        if path.endswith('.py'):
            linter.check_file(path)
            linter.print_report()
        else:
            # For non-Python files (like your pasted code snippet)
            with open(path, 'r') as f:
                content = f.read()
            
            # Check for specific issues in important functions
            specific_issues = find_specific_issues(content)
            if specific_issues:
                print(f"\nFound {len(specific_issues)} potential issues in key functions:")
                for issue in specific_issues:
                    print(f"Line {issue['line']} in function {issue['function']}: {issue['message']}")
            else:
                print("No specific issues found in key functions.")


if __name__ == "__main__":
    main()
