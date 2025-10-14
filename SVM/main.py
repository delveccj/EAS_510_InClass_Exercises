#!/usr/bin/env python3
"""
Main script to run SVM examples.
Usage: python main.py [example_type]
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_environment
from linear_svm import LinearSVMExamples
from nonlinear_svm import NonlinearSVMExamples
from soft_margin import SoftMarginSVMExamples
from svm_regression import SVMRegressionExamples

def show_menu():
    """Show interactive menu for selecting SVM examples."""
    print("🎯 SVM Examples - Standalone Python Implementation")
    print("=" * 50)
    print("Select which examples to run:")
    print("1. Linear SVM Examples")
    print("2. Soft Margin SVM Examples") 
    print("3. Nonlinear SVM Examples")
    print("4. SVM Regression Examples")
    print("5. All Examples")
    print("0. Exit")
    print("-" * 50)
    
    while True:
        try:
            choice = input("Enter your choice (0-5): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5']:
                return choice
            else:
                print("Invalid choice. Please enter a number between 0-5.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return '0'

def run_examples(choice):
    """Run the selected SVM examples."""
    setup_environment()
    
    if choice == '1':
        print("\n📏 Linear SVM Examples:")
        linear_examples = LinearSVMExamples()
        linear_examples.run_all()
    
    elif choice == '2':
        print("\n🎛️ Soft Margin SVM Examples:")
        soft_examples = SoftMarginSVMExamples()
        soft_examples.run_all()
    
    elif choice == '3':
        print("\n🎪 Nonlinear SVM Examples:")
        nonlinear_examples = NonlinearSVMExamples()
        nonlinear_examples.run_all()
    
    elif choice == '4':
        print("\n📊 SVM Regression Examples:")
        regression_examples = SVMRegressionExamples()
        regression_examples.run_all()
    
    elif choice == '5':
        print("\n📏 Linear SVM Examples:")
        linear_examples = LinearSVMExamples()
        linear_examples.run_all()
        
        print("\n🎛️ Soft Margin SVM Examples:")
        soft_examples = SoftMarginSVMExamples()
        soft_examples.run_all()
        
        print("\n🎪 Nonlinear SVM Examples:")
        nonlinear_examples = NonlinearSVMExamples()
        nonlinear_examples.run_all()
        
        print("\n📊 SVM Regression Examples:")
        regression_examples = SVMRegressionExamples()
        regression_examples.run_all()
    
    print("\n✅ Examples completed!")

def main():
    """Run SVM examples with command line args or interactive menu."""
    # Check for command line arguments first
    if len(sys.argv) > 1:
        example_type = sys.argv[1].lower()
        setup_environment()
        
        print("🎯 SVM Examples - Standalone Python Implementation")
        print("=" * 50)
        
        if example_type in ["linear", "1"]:
            print("\n📏 Linear SVM Examples:")
            linear_examples = LinearSVMExamples()
            linear_examples.run_all()
        
        elif example_type in ["soft", "2"]:
            print("\n🎛️ Soft Margin SVM Examples:")
            soft_examples = SoftMarginSVMExamples()
            soft_examples.run_all()
        
        elif example_type in ["nonlinear", "3"]:
            print("\n🎪 Nonlinear SVM Examples:")
            nonlinear_examples = NonlinearSVMExamples()
            nonlinear_examples.run_all()
        
        elif example_type in ["regression", "4"]:
            print("\n📊 SVM Regression Examples:")
            regression_examples = SVMRegressionExamples()
            regression_examples.run_all()
        
        elif example_type in ["all", "5"]:
            run_examples('5')
            return
        
        else:
            print(f"Unknown example type: {example_type}")
            print("Available options: linear, soft, nonlinear, regression, all")
            return
        
        print("\n✅ Examples completed!")
    
    else:
        # Interactive mode
        while True:
            choice = show_menu()
            
            if choice == '0':
                print("Goodbye! 👋")
                break
            
            run_examples(choice)
            
            # Ask if user wants to run more examples
            print("\n" + "-" * 50)
            try:
                again = input("Run more examples? (y/n): ").strip().lower()
                if again not in ['y', 'yes']:
                    print("Goodbye! 👋")
                    break
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break

if __name__ == "__main__":
    main()