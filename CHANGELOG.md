# Changelog

All notable changes to ABACUS-STRU-Analyser will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-08-27

### ✨ Added

#### PCA Dimensionality Reduction
- **New `--pca_components` parameter**: Configurable PCA components (default: 3)
- **PCA-based analysis**: All computations performed in reduced dimensional space
- **Automatic padding**: Zero-padding when original dimensions < requested PCA components
- **PCA components output**: `pca_components.csv` with per-frame principal component values
- **Scikit-learn integration**: Added scikit-learn dependency for PCA functionality

#### Enhanced Analysis Pipeline
- **Unified PCA workflow**: Consistent PCA application across single and batch analysis
- **Parallel processing support**: PCA parameters propagated to worker processes
- **Result saving**: Integrated PCA data saving with existing result management

### 🔧 Changed

#### Dependencies
- **Added scikit-learn**: Required for PCA dimensionality reduction
- **Updated requirements.txt**: Includes scikit-learn>=1.3

#### Configuration
- **New command-line option**: `--pca_components` with default value 3
- **Enhanced SystemAnalyzer**: Added `pca_components` parameter to constructor

### 📚 Documentation

#### User Documentation
- **README.md updated**: Added PCA feature description and usage
- **Parameter documentation**: Updated configuration table with PCA options
- **Output files**: Added `pca_components.csv` to output file list

#### Technical Documentation
- **Implementation details**: PCA reduction and component extraction methods
- **Error handling**: Scikit-learn import error handling and user guidance

### 🔒 Security

- **Dependency validation**: Proper error handling for missing scikit-learn

## [2.0.0] - 2025-08-21

### 🎉 Major Release - Breaking Changes

This is a complete architectural rewrite focusing on code quality, maintainability, and developer experience.

### ✨ Added

#### New Architecture
- **Modular Utils Package**: Split monolithic `utils.py` into specialized modules:
  - `src/utils/data_utils.py` - Data validation and processing utilities
  - `src/utils/file_utils.py` - File and directory operations
  - `src/logging/manager.py` - Centralized logging management
- **Enhanced Type Safety**: Added comprehensive type annotations throughout the codebase
- **Centralized Logging**: New `LoggerManager` class for consistent logging across all modules

#### Development Infrastructure
- **Testing Framework**: Complete test suite with pytest
  - Unit tests for all utility modules
  - Integration tests for core functionality
  - Smoke tests for end-to-end validation
- **Code Quality Tools**: Integrated development toolchain
  - `ruff` for fast Python linting
  - `black` for code formatting
  - `mypy` for static type checking
  - `pytest` with coverage reporting
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing
- **Pre-commit Hooks**: Optional git hooks for code quality enforcement

#### Migration Tools
- **Migration Script**: `tools/migration/migrate_v1_to_v2.py` for automated data migration
- **Upgrade Documentation**: Comprehensive upgrade guide with step-by-step instructions
- **Backward Compatibility**: Legacy aliases and compatibility layer for smooth transition

### 🔧 Changed

#### Breaking Changes
- **Import Paths**: Some utility classes moved to specialized modules
  ```python
  # v1.x
  from src.utils import LoggerManager
  
  # v2.0
  from src.logging import LoggerManager
  ```
- **Exception Handling**: Replaced bare `except:` clauses with specific exception handling
- **Logging Configuration**: Removed module-level `logging.basicConfig()` calls
- **File Structure**: Reorganized codebase into logical modules

#### Improvements
- **Error Handling**: Enhanced exception handling with detailed error messages and stack traces
- **Path Management**: Improved file path handling and directory management
- **Result Saving**: More robust CSV writing with better error recovery
- **Code Organization**: Clear separation of concerns across modules

### 🐛 Fixed

#### Code Quality Issues
- **Empty Data Handling**: Unified empty checking across numpy arrays and regular lists
- **Exception Silencing**: Replaced silent exception handling with proper logging
- **Import Dependencies**: Fixed circular imports and undefined references
- **Memory Management**: Better resource cleanup in file operations

#### Functionality Fixes
- **CSV Processing**: Improved handling of malformed CSV files
- **Directory Creation**: More robust directory creation with proper error handling
- **Log File Management**: Better log file rotation and cleanup

### 🚀 Performance

- **Import Optimization**: Reduced startup time through optimized imports
- **Memory Usage**: Better memory management in large dataset processing
- **Error Recovery**: Faster error recovery without silent failures

### 📚 Documentation

- **API Documentation**: Enhanced docstrings with type information
- **Migration Guide**: Complete upgrade documentation with examples
- **Development Setup**: Detailed instructions for development environment
- **Testing Guide**: Documentation for running and extending tests

### 🔒 Security

- **Input Validation**: Enhanced validation of user inputs and file paths
- **Error Information**: Careful handling of sensitive information in error messages
- **File Operations**: Safer file operations with proper permission checking

### 💥 Breaking Changes Summary

1. **Module Reorganization**:
   - Logging utilities moved to `src.logging` package
   - File utilities enhanced in `src.utils.file_utils`
   - Data utilities enhanced in `src.utils.data_utils`

2. **API Changes**:
   - `LoggerManager` import path changed
   - Some utility functions have enhanced signatures
   - Exception handling behavior changed (no more silent failures)

3. **Configuration**:
   - Logging configuration now centralized
   - Some default file paths may be different
   - Enhanced configuration validation

4. **Dependencies**:
   - Added development dependencies for code quality tools
   - Updated minimum Python version requirement (3.8+)

### 🔄 Migration Path

For users upgrading from v1.x:

1. **Automated Migration**: Use `tools/migration/migrate_v1_to_v2.py`
2. **Manual Changes**: Update import statements as needed
3. **Testing**: Run the test suite to verify functionality
4. **Documentation**: Review `UPGRADE_TO_V2.md` for detailed instructions

### 📦 Dependencies

#### Added
- `ruff>=0.1.0` - Fast Python linter
- `black>=23.0.0` - Code formatter
- `mypy>=1.5.0` - Static type checker
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting

#### Updated
- Enhanced `requirements.txt` with pinned versions
- Added `requirements-dev.txt` for development dependencies

---

## [1.x] - Previous Versions

Previous versions were focused on core functionality development. See git history for detailed changes in v1.x releases.

---

### Migration Notes

- **Backup Required**: Always backup your data before upgrading to v2.0
- **Testing Recommended**: Use the migration script's dry-run mode first
- **Support Available**: See `UPGRADE_TO_V2.md` for detailed migration assistance

### Coming Next

Future versions will focus on:
- Additional analysis algorithms
- Performance optimizations
- Enhanced visualization capabilities
- Extended format support
