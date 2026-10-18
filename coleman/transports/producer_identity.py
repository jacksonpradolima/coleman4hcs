"""Compatible import path for the shared transport-independent package identity.

Existing optional capture APIs retain their original function names and digest
algorithm. Package resources/version remain part of the identity, so a new SDK
still requires a new producer generation; old captures retain their old package.
"""

from coleman.spec.producer_identity import installed_producer_identity, package_pins

__all__ = ["installed_producer_identity", "package_pins"]
