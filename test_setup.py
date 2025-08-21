#!/usr/bin/env python3
"""
Test script to verify Business Optima setup
Run this script to check if all dependencies are properly installed
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'flask',
        'PyPDF2',
        'sentence_transformers',
        'faiss',
        'numpy',
        'sklearn',
 ""        'reportlab'
    ]
    
    print("🔍 Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_python_version():
    """Test Python version compatibility."""
    print(f"\n🐍 Python version: {sys.version}")
    
    if sys.version_info >= (3, 13):
        print("✅ Python 3.13+ detected - Compatible")
        return True
    else:
        print("❌ Python 3.13+ required")
        return False

def test_flask_app():
    """Test if Flask app can be created."""
    try:
        from app import app
        print("✅ Flask app can be imported")
        return True
    except Exception as e:
        print(f"❌ Flask app import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Business Optima Setup Test")
    print("=" * 40)
    
    # Test Python version
    python_ok = test_python_version()
    
    # Test package imports
    failed_imports = test_imports()
    
    # Test Flask app
    flask_ok = test_flask_app()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    if python_ok and not failed_imports and flask_ok:
        print("🎉 All tests passed! Your setup is ready.")
        print("\nTo run the application:")
        print("  python app.py")
        print("\nTo deploy to Railway:")
        print("  railway up")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        
        if failed_imports:
            print(f"\n📦 Install missing packages:")
            print(f"  pip install {' '.join(failed_imports)}")
        
        if not python_ok:
            print("\n🐍 Upgrade to Python 3.13+")
        
        if not flask_ok:
            print("\n🔧 Check app.py for syntax errors")

if __name__ == "__main__":
    main()
