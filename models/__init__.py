from .base import *
from .vanilla_vae import *
from .beta_vae import *
from .vanilla_vaetest import *

# Aliases
VAE = VanillaVAE

vae_models = {
              'BetaVAE':BetaVAE,
              'VanillaVAE':VanillaVAE,
              'VanillaVAETest': VanillaVAETest}
