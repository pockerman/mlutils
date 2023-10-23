from pathlib import Path
import torch.nn as nn
import torch
from src.easyfsl.methods.prototypical_networks import PrototypicalNetworks


class EasyFSLProtoNetClassifier(PrototypicalNetworks):
    """Wrapper to easyfsl PrototypicalNetwork

    """

    def __init__(self, backbone: nn.Module, model_name: str, device: str):
        super(EasyFSLProtoNetClassifier, self).__init__(backbone)
        self.name = model_name
        self.device = device

    def load_prototypes(self, prototypes_paths: Path, device: str) -> None:
        """Load the prototypes from the given paths

        Parameters
        ----------
        device
        prototypes_paths

        Returns
        -------

        """
        self.set_prototypes(prototypes=torch.load(prototypes_paths,
                                                  map_location=torch.device(device)))
        self.device = device

    def set_prototypes(self, prototypes: torch.Tensor) -> None:
        """Set the prototypes for the network. Is assumes that
        the data is already mapped in the proper device

        Parameters
        ----------
        prototypes: The prototypes for the network

        Returns
        -------

        """
        self.prototypes = prototypes

    def save_prototypes(self, filename: Path):

        if isinstance(self.prototypes, torch.Tensor):
            torch.save(self.prototypes, str(filename))
        else:
            raise ValueError("Model prototypes are not an instance of torch.Tensor")
