# Style guide
## Naming conventions
**Language used :**
English 

**Case type :**

- PascalCase for classes. 

```python
    class MyClass:
```
- snake_case for files, folders, functions and variables.

```python
    def train_model():
```
- UPPER_SNAKE_CASE for configuration variables.

```python
    RANDOM_STATE = 42
```

**Format :**
- All functions should follow the "verb_object" format. 
- Folder names should be kept to one single word. 


## Pytest conventions
**Test functions and files naming:** Test functions and files must start with `test_` .

Example : `test_clean_data.py` .

**Class naming:** If tests are grouped in a class, the class must start with `Test` .


```python 
class TestPreprocessing:
```
**Structure:**
Our tests folder will mirror our src directory.

```
├── src/
   └── data_processing.py
...
└── tests/
    └── test_data_processing.py
```

## Code documentation

**Style:** Google Docstring Format

```
"""Summary of the function.

Detailed description of the function, if necessary. The summary line should
be a concise statement of what the function does.

Args:
    param1 (type): Description of the first parameter.
    param2 (type, optional): Description of the second parameter.
        Defaults to None.

Returns:
    type: Description of the return value.

Raises:
    ExceptionType: Description of the exception that can be raised.
"""
```

**Requirement:** All public functions and classes must include docstrings.

## Git branches 

**Case type:**
- snake-case for branche names.
- The branch name should be descriptive and concise.

**Format:**
- Every branch should start with a Prefixe. 
Example : `prefixe/branche-name`
- The prefixe helps to quickly identify the purpose of the branche. 
Example : 
    - `feature/`: for developing new features.
    - `bugfix/`: to fix bugs in the code.
    - `docs/`: To write or update documentation.
- Only documentation is allowed to be part of feature and bugfix branches. It is not recommeneded to fix a bug while adding a feature, nor add a feature while fixing a bug.