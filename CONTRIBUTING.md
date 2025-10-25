# Contributing to E-Commerce Data Analysis Project

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

- Be respectful and constructive in all interactions
- Welcome diverse perspectives and experiences
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

Before creating bug reports, check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**
- **Include your Python version and OS**

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and the expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow the Python style guide
- Include appropriate test cases
- End all files with a newline
- Avoid platform-dependent code
- Place imports in the following groups: stdlib, related third party, local
- Use absolute imports

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment
3. Install dependencies: `pip install -r requirements.txt`
4. Create a feature branch: `git checkout -b feature/YourFeature`
5. Make your changes
6. Test your changes: `python your_script.py`
7. Commit with clear messages: `git commit -m "Add YourFeature"`
8. Push to your fork: `git push origin feature/YourFeature`
9. Submit a Pull Request

## Style Guide

### Python Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep lines under 100 characters
- Use 4 spaces for indentation

Example:
```python
def calculate_metrics(data):
    """
    Calculate key performance metrics.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data with required columns
        
    Returns
    -------
    dict
        Dictionary containing calculated metrics
    """
    metrics = {}
    metrics['mean'] = data['value'].mean()
    metrics['std'] = data['value'].std()
    return metrics
```

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line
- When only changing documentation, include `[ci skip]`

Example:
```
Add anomaly detection visualization

This adds a new chart for detecting anomalies using Z-score analysis.
It helps identify outliers in the revenue data.

Fixes #123
```

## Testing

- Test your changes before submitting
- Ensure your changes don't break existing functionality
- Add test cases for new features
- Run the full analysis to verify output

## Documentation

- Update README if you change functionality
- Add docstrings to new functions
- Update CHANGELOG for significant changes
- Keep documentation up-to-date with code changes

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- GitHub contributors page

## Questions?

Feel free to ask questions by:
- Opening an issue with the question tag
- Contacting the project maintainer
- Joining our community discussions

Thank you for contributing!
