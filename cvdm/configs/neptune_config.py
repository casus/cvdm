from dataclasses import dataclass


@dataclass
class NeptuneConfig:
    """
    Configuration settings for logging and tracking experiments with Neptune. Can be omitted if Neptune should not be used.

    Attributes:
        name (str): The name of te run to be logged in Neptune.
        project (str): The Neptune project identifier where the experiment will be tracked, typically in the format 'workspace/project-name'.
    """

    name: str
    project: str
