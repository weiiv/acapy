"""Registry of named `Signer` implementations for external signing.

Plugins register a signer at startup; `BaseWallet._get_external_signature`
dispatches to the one named in the DID's `signer` metadata.
"""

from typing import Dict, List, Optional, Protocol, Union, runtime_checkable

from .key_type import KeyType


@runtime_checkable
class Signer(Protocol):
    """Signer protocol implemented by external-signer plugins."""

    async def sign(
        self, key_ref: str, message: Union[List[bytes], bytes], key_type: KeyType
    ) -> bytes:
        """Sign `message` with the key identified by `key_ref`.

        `message` is passed through unchanged; a list only arises for
        multi-message schemes such as BBS+. Return the raw signature in
        the same encoding the wallet would produce for `key_type`.
        """
        ...


class SignerRegistry:
    """Registry of `Signer` implementations keyed by provider name."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._signers: Dict[str, Signer] = {}

    def register(self, name: str, signer: Signer) -> None:
        """Register `signer` under `name`. Raises ValueError on duplicate."""
        if name in self._signers:
            raise ValueError(f"Signer already registered as {name}")
        self._signers[name] = signer

    def get(self, name: str) -> Optional[Signer]:
        """Return the signer registered under `name`, or None if not found."""
        return self._signers.get(name)

    def names(self) -> List[str]:
        """Return all registered signer names."""
        return list(self._signers.keys())
