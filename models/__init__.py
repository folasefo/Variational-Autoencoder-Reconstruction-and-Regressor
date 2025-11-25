from .base import *
from .vanilla_vae import *
from .gamma_vae import *
from .beta_vae import *
from .cvae import *
from .hvae import *
from .iwae import *
from .dfcvae import *
from .mssim_vae import MSSIMVAE
from .fvae import *
from .cat_vae import *
from .joint_vae import *
from .info_vae import *
# from .twostage_vae import *
from .lvae import LVAE
from .logcosh_vae import *
from .swae import *
from .miwae import *
from .betatc_vae import *
from .dip_vae import *

from .vanilla_vaetest import *


# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE
CVAE = ConditionalVAE
GumbelVAE = CategoricalVAE

vae_models = {'HVAE':HVAE,
              'LVAE':LVAE,
              'IWAE':IWAE,
              'SWAE':SWAE,
              'MIWAE':MIWAE,
              'DFCVAE':DFCVAE,
              'DIPVAE':DIPVAE,
              'BetaVAE':BetaVAE,
              'InfoVAE':InfoVAE,
              'GammaVAE':GammaVAE,
              'MSSIMVAE':MSSIMVAE,
              'JointVAE':JointVAE,
              'BetaTCVAE':BetaTCVAE,
              'FactorVAE':FactorVAE,
              'LogCoshVAE':LogCoshVAE,
              'VanillaVAE':VanillaVAE,
              'ConditionalVAE':ConditionalVAE,
              'CategoricalVAE':CategoricalVAE,
              'VanillaVAETest': VanillaVAETest}
