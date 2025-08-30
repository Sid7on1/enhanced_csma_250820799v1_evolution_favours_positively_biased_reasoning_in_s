import logging
import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UtilityFunctions:
    """
    A class containing various utility functions for the agent project.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the utility functions with a configuration dictionary.

        Args:
        - config (Dict[str, Any]): A dictionary containing configuration settings.
        """
        self.config = config

    def validate_input(self, data: Any) -> bool:
        """
        Validate the input data.

        Args:
        - data (Any): The input data to be validated.

        Returns:
        - bool: True if the input data is valid, False otherwise.
        """
        try:
            # Check if the input data is of the correct type
            if not isinstance(data, (int, float, str, list, dict, np.ndarray, pd.DataFrame, torch.Tensor)):
                logger.error("Invalid input type")
                return False

            # Check if the input data is within the valid range
            if isinstance(data, (int, float)) and (data < self.config["min_value"] or data > self.config["max_value"]):
                logger.error("Input value out of range")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating input: {str(e)}")
            return False

    def calculate_velocity_threshold(self, data: np.ndarray) -> float:
        """
        Calculate the velocity threshold based on the input data.

        Args:
        - data (np.ndarray): The input data.

        Returns:
        - float: The calculated velocity threshold.
        """
        try:
            # Calculate the velocity threshold using the formula from the paper
            velocity_threshold = np.mean(data) + self.config["velocity_threshold_std_dev"] * np.std(data)
            return velocity_threshold

        except Exception as e:
            logger.error(f"Error calculating velocity threshold: {str(e)}")
            return None

    def apply_flow_theory(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the flow theory to the input data.

        Args:
        - data (np.ndarray): The input data.

        Returns:
        - np.ndarray: The output data after applying the flow theory.
        """
        try:
            # Apply the flow theory using the formula from the paper
            output_data = data + self.config["flow_theory_coefficient"] * np.gradient(data)
            return output_data

        except Exception as e:
            logger.error(f"Error applying flow theory: {str(e)}")
            return None

    def calculate_metrics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate various metrics based on the input data.

        Args:
        - data (np.ndarray): The input data.

        Returns:
        - Dict[str, float]: A dictionary containing the calculated metrics.
        """
        try:
            # Calculate the metrics using the formulas from the paper
            metrics = {
                "mean": np.mean(data),
                "std_dev": np.std(data),
                "velocity_threshold": self.calculate_velocity_threshold(data),
            }
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return None

    def save_data(self, data: Any, filename: str) -> bool:
        """
        Save the input data to a file.

        Args:
        - data (Any): The input data to be saved.
        - filename (str): The filename to save the data to.

        Returns:
        - bool: True if the data is saved successfully, False otherwise.
        """
        try:
            # Save the data to a file using the appropriate method
            if isinstance(data, np.ndarray):
                np.save(filename, data)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(filename, index=False)
            elif isinstance(data, torch.Tensor):
                torch.save(data, filename)
            else:
                with open(filename, "w") as f:
                    f.write(str(data))

            return True

        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False

    def load_data(self, filename: str) -> Any:
        """
        Load data from a file.

        Args:
        - filename (str): The filename to load the data from.

        Returns:
        - Any: The loaded data.
        """
        try:
            # Load the data from a file using the appropriate method
            if filename.endswith(".npy"):
                data = np.load(filename)
            elif filename.endswith(".csv"):
                data = pd.read_csv(filename)
            elif filename.endswith(".pt"):
                data = torch.load(filename)
            else:
                with open(filename, "r") as f:
                    data = f.read()

            return data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None

class Configuration:
    """
    A class containing configuration settings.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration settings.

        Args:
        - config (Dict[str, Any]): A dictionary containing configuration settings.
        """
        self.config = config

    def get_config(self, key: str) -> Any:
        """
        Get a configuration setting.

        Args:
        - key (str): The key of the configuration setting.

        Returns:
        - Any: The value of the configuration setting.
        """
        try:
            return self.config[key]

        except Exception as e:
            logger.error(f"Error getting config: {str(e)}")
            return None

class ExceptionClasses:
    """
    A class containing custom exception classes.
    """

    class InvalidInputError(Exception):
        """
        A custom exception class for invalid input.
        """

        def __init__(self, message: str):
            """
            Initialize the exception.

            Args:
            - message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

    class DataLoadingError(Exception):
        """
        A custom exception class for data loading errors.
        """

        def __init__(self, message: str):
            """
            Initialize the exception.

            Args:
            - message (str): The error message.
            """
            self.message = message
            super().__init__(self.message)

def main():
    # Create a configuration dictionary
    config = {
        "min_value": 0,
        "max_value": 100,
        "velocity_threshold_std_dev": 1.0,
        "flow_theory_coefficient": 0.5,
    }

    # Create an instance of the utility functions class
    utility_functions = UtilityFunctions(config)

    # Test the utility functions
    data = np.array([1, 2, 3, 4, 5])
    velocity_threshold = utility_functions.calculate_velocity_threshold(data)
    output_data = utility_functions.apply_flow_theory(data)
    metrics = utility_functions.calculate_metrics(data)

    logger.info(f"Velocity threshold: {velocity_threshold}")
    logger.info(f"Output data: {output_data}")
    logger.info(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()